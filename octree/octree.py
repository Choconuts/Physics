import time
import numpy as np
import torch
import matplotlib.pyplot as plt


DEBUG_OCTREE = False


def valid_box(boxes):
    """
    [..., 6] -> [..., 1]
    """
    return torch.prod(boxes[..., 3:] > 0, -1, keepdim=True)


def into_box(boxes, x, inv=False):
    """
    [..., 6] X [..., 3] -> [..., 3]
    """
    if inv:
        return x * boxes[..., 3:] + boxes[..., :3]
    return (x - boxes[..., :3]) / boxes[..., 3:]


def inside_box(boxes, x, exactly=True):
    """
    [..., 6] X [..., 3] -> [..., 1]
    """
    if not exactly:
        lt1 = torch.prod(into_box(boxes, x) <= 1, -1, keepdim=True)
        gt0 = torch.prod(into_box(boxes, x) >= 0, -1, keepdim=True)
        return lt1 * gt0
    lt1 = torch.prod(into_box(boxes, x) < 1, -1, keepdim=True)
    gt0 = torch.prod(into_box(boxes, x) > 0, -1, keepdim=True)
    return lt1 * gt0


def trunc_div(a, b):
    return torch.div(a, b, rounding_mode='trunc')


def bin_to_oct(x, inv=False):
    """
    [..., 3] -> [..., 1]
    """
    if inv:
        return torch.cat([trunc_div(x, 4) % 2, trunc_div(x, 2) % 2, x % 2], -1)
    return 4 * x[..., 0] + 2 * x[..., 1] + x[..., 2]


def which_oct_cell(boxes, x):
    """
    [..., 6] X [..., 3] -> [..., 1]
    """
    idx = (into_box(boxes, x) * 2).long()
    idx = torch.clip(idx, 0, 1)
    return bin_to_oct(idx)


def intersect_box(boxes, rays_o, rays_d, forward_only=True):
    """
    [N?, 3] X [N, 3], [N, 3] -> [N, 1], [N, 1], [N, 1]
    """
    inv_dir = 1.0 / rays_d
    t_min = (boxes[..., :3] - rays_o) * inv_dir
    t_max = (boxes[..., 3:] + boxes[..., :3] - rays_o) * inv_dir
    t1 = torch.minimum(t_min, t_max)
    t2 = torch.maximum(t_min, t_max)

    near = torch.maximum(torch.maximum(t1[..., 0:1], t1[..., 1:2]), t1[..., 2:3])
    far = torch.minimum(torch.minimum(t2[..., 0:1], t2[..., 1:2]), t2[..., 2:3])

    if forward_only:
        return torch.logical_and(near <= far, far >= 0), torch.maximum(near, torch.zeros_like(near)), far

    return near <= far, near, far


def divide(boxes):
    """
    [..., 6] -> [..., 8, 6]
    """
    box_min = boxes[..., None, :3]
    box_size = boxes[..., None, 3:]
    ofs = bin_to_oct(torch.linspace(0, 7, 8, dtype=torch.long, device=boxes.device).view(-1, 1), True)       # [8, 3]
    new_min = box_min + ofs * box_size / 2
    new_size = (box_size / 2).expand(new_min.shape)

    return torch.cat([new_min, new_size], -1)


class Octree:

    def __init__(self, box_min, box_size, device="cuda"):
        self.boxes = torch.cat([torch.tensor(box_min), torch.tensor(box_size)]).view(1, 6)
        self.non_leaf = torch.zeros(1, 1, dtype=torch.bool)
        self.links = torch.zeros(1, 8, dtype=torch.long)
        self.device = device
        if device == "cuda":
            self.cuda()
        self.max_depth = -1
        self.cache_index = None

    def cuda(self):
        self.device = "cuda"
        self.boxes = self.boxes.cuda()
        self.non_leaf = self.non_leaf.cuda()
        self.links = self.links.cuda()
        return self

    def add_nodes(self, number):
        new_boxes = torch.zeros(number, 6, device=self.device)
        new_non_leaf = torch.zeros(number, 1, dtype=torch.long, device=self.device)
        new_links = torch.zeros(number, 8, dtype=torch.long, device=self.device)
        self.boxes = torch.cat([self.boxes, new_boxes], 0)
        self.non_leaf = torch.cat([self.non_leaf, new_non_leaf], 0)
        self.links = torch.cat([self.links, new_links], 0)

    def build(self, div_fn, max_depth, min_depth):
        starts = [0]
        ends = [1]

        for i in range(max_depth):
            s = starts[i]
            e = ends[i]
            boxes = self.boxes[s:e]
            if i < min_depth:
                dived = torch.ones_like(boxes[..., 0:1], dtype=torch.long)
            else:
                dived = div_fn(boxes)
            k = dived.nonzero()[..., 0]
            n = k.shape[0]
            self.non_leaf[s:e] = dived.view(-1, 1)
            lsp1 = torch.linspace(0, 7, 8, dtype=torch.long, device=self.device)
            lsp2 = torch.linspace(e, e + (n - 1) * 8, n, dtype=torch.long, device=self.device)
            new_links = lsp1[None, :] + lsp2[:, None]
            self.links[s:e][k] = new_links
            self.add_nodes(n * 8)

            self.boxes[e:] = divide(boxes[k]).view(-1, 6)
            starts += [e]
            ends += [e + n * 8]

        if DEBUG_OCTREE:
            assert valid_box(self.boxes).all()
            assert self.boxes.shape[0] == self.links.shape[0] == self.non_leaf.shape[0]

        self.combine_empty(max_depth)
        self.cache_index = self.gen_grid_index(max(min_depth, 6))

        bytes = (self.boxes.numel() + self.boxes.numel() + self.non_leaf.numel() + self.cache_index.numel()) * 4
        print(self.boxes.shape[0], "boxes", bytes // 1024 // 1024, "MB")

    def combine_empty(self, max_depth):
        leaf_size = self.boxes[0, 3:] / (2 ** max_depth)

        def is_cell(bosex):
            return (bosex[..., 3:] < leaf_size + 1e-4).prod(-1).bool()

        has_cell_child = is_cell(self.boxes)
        non_leaf = self.non_leaf[..., 0].bool()

        for i in range(max_depth):
            ptr = self.links[non_leaf]
            not_empty = has_cell_child[ptr].sum(-1).bool()
            has_cell_child[non_leaf] = not_empty

        self.non_leaf[non_leaf] = has_cell_child.long()[..., None][non_leaf]
        self.max_depth = max_depth

    def gen_grid_index(self, depth):
        res = 2 ** depth
        lsp = torch.linspace(0, 1.0, res + 1, device=self.boxes.device)[:-1]
        anchor = torch.stack(torch.meshgrid([lsp, lsp, lsp], indexing="ij"), -1).view(-1, 3)
        anchor = anchor + 0.5 / res
        anchor = into_box(self.boxes[0], anchor, inv=True)
        index = self.query(anchor, depth)
        return index.view(res, res, res)

    def get_cache(self, x):
        min_res = self.cache_index.shape
        local_x = into_box(self.boxes[0], x)
        idx = torch.split((local_x * torch.tensor(min_res, device=local_x.device)).floor().long(), 1, -1)
        ptr = self.cache_index[idx][..., 0]
        return ptr

    def query(self, x, max_depth=-1, no_cache=False):
        """
        [..., 3] -> [..., 1]
        """
        init_shape = list(x.shape[:-1]) + [-1]
        x = x.view(-1, 3)

        inside_root = inside_box(self.boxes[0], x).bool()[..., 0]
        all_ptr = -torch.ones_like(x[..., 0], dtype=torch.long)
        x = x[inside_root]

        if x.numel() == 0:
            return all_ptr

        if not no_cache and self.cache_index is not None:
            min_res = self.cache_index.shape
            local_x = into_box(self.boxes[0], x)
            idx = torch.split((local_x * torch.tensor(min_res, device=local_x.device)).floor().long(), 1, -1)
            ptr = self.cache_index[idx][..., 0]

            # assert (self.query(x, 6, True)[..., 0] == ptr).all()

        else:
            ptr = torch.zeros_like(x[..., 0], dtype=torch.long)

        if DEBUG_OCTREE:
            assert inside_box(self.boxes[0], x).all()

        k = self.non_leaf[ptr].nonzero()[..., 0]

        while k.numel() > 0:
            if max_depth == 0:
                break
            max_depth -= 1

            if DEBUG_OCTREE:
                assert (k < ptr.shape[0]).all() and (k >= 0).all()
                assert (ptr < self.links.shape[0]).all() and (ptr >= 0).all()
                assert x.shape[0] == ptr.shape[0]

            boxes = self.boxes[ptr[k]]
            # assert inside_box(boxes, x[k]).all()
            oct_idx = which_oct_cell(boxes, x[k])
            sub_links = torch.gather(self.links[ptr[k]], -1, oct_idx[:, None])[..., 0]
            ptr[k] = sub_links
            k = self.non_leaf[ptr].nonzero()[..., 0]

        all_ptr[inside_root] = ptr
        return all_ptr.view(init_shape)

    def relative_query(self, x, ptrs=None):
        """
            [N, 3] x [N, 1] -> [N, 1] (move ptr)
        """
        if ptrs is None:
            ptrs = Stack(x.shape[0], self.max_depth + 1)
            ptrs.push(torch.zeros_like(x[..., 0], dtype=torch.long))
        else:
            inside = inside_box(self.boxes[ptrs.peek()], x)[..., 0]
            while (~inside).any():
                ptrs.pop(~inside)
                inside = inside_box(self.boxes[ptrs.peek()], x)[..., 0]

        k = self.non_leaf[ptrs.peek()].bool()[..., 0]

        while k.any():
            boxes = self.boxes[ptrs.peek(k)]
            oct_idx = which_oct_cell(boxes, x[k])
            sub_links = torch.gather(self.links[ptrs.peek(k)], -1, oct_idx[:, None])[..., 0]
            ptrs.push(sub_links, k)
            k = self.non_leaf[ptrs.peek()].bool()[..., 0]

        return ptrs

    def cast(self, rays_o, rays_d, hit_fn, eps=5e-4):
        """
            [..., 3], [..., ?, 3] -> first hit box satisfying hit_fn (leaf only)
        """
        init_shape = list(rays_o.shape[:-1]) + [-1]
        if len(rays_d.shape) < len(rays_o.shape):
            rays_d = rays_d[..., None, :]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.expand(init_shape).reshape(-1, 3)

        # assume all intersect
        valid, near, far = intersect_box(self.boxes[0], rays_o, rays_d)

        k = valid[..., 0]
        t = near + eps
        t[~k] = -1

        ptr = -torch.ones_like(k, dtype=torch.long)
        pos = torch.zeros_like(rays_o)
        pos[k] = rays_o[k] + t[k] * rays_d[k]
        ptr[k] = self.query(pos[k])[..., 0]
        k = ptr >= 0

        st = time.time()
        ii = 0
        while k.any():
            valid, near, far = intersect_box(self.boxes[ptr[k]], pos[k], rays_d[k])

            if DEBUG_OCTREE:
                if (near != 0).any():
                    tmp = near != 0
                    assert (near[tmp].abs() < 1e-4).all()
                    near[tmp] = 0
                if (~valid).any():
                    assert (far[~valid[..., 0]].abs() < 1e-4).all()
                    far[~valid[..., 0]] = near[~valid[..., 0]]

            t[k] = t[k] + far - near + eps
            pos[k] = rays_o[k] + t[k] * rays_d[k]
            ptr[k] = self.query(pos[k])[..., 0]
            k = ptr > 0

            ii += 1

            if k.any():
                hit = hit_fn(self.boxes[ptr[k]]).bool()
                if len(hit.shape) == 2:
                    hit = hit[..., 0]
                k[k.clone()] = ~hit

        return t.reshape(init_shape)


class Stack:

    def __init__(self, n, cap, device="cuda"):
        self.stack = -torch.ones(n, cap + 1, device=device, dtype=torch.long)
        self.ptr = torch.zeros(n, dtype=torch.long, device=device)
        self.mask = None

    def make_mask(self, mask):
        if mask is None:
            if self.mask is None:
                mask = torch.ones_like(self.ptr).bool()
            else:
                mask = self.mask
        elif self.mask is not None:
            mask = torch.where(self.mask, mask, self.mask)
        return mask

    def push(self, values, mask=None):
        mask = self.make_mask(mask)
        self.ptr[mask] = self.ptr[mask] + 1
        self.stack[mask, self.ptr[mask]] = values
        assert self.ptr.max() < self.stack.shape[-1]

    def peek(self, mask=None):
        mask = self.make_mask(mask)
        return torch.gather(self.stack[mask], -1, self.ptr[mask][..., None])[..., 0]

    def pop(self, mask=None):
        mask = self.make_mask(mask)
        self.stack[mask, self.ptr[mask]] = -1
        self.ptr[mask] = self.ptr[mask] - 1
        assert self.ptr.min() >= 0

    def masked(self, mask):
        self.mask = mask
        return self

    def __repr__(self):
        return str(self.stack) + "\n" + str(self.ptr)


if __name__ == '__main__':
    torch.random.manual_seed(0)
    np.random.seed(0)

    boxes = [[0, 0, 0, 2, 2, 2], [1, 1, 1, 2, 2, 2.]]
    rays_o = [[4, 4, 4], [3, 3, 4.]]
    rays_d = [[-1, -1, -1], [-1, -1, -1.]]
    v, n, f = intersect_box(torch.tensor(boxes), torch.tensor(rays_o), torch.tensor(rays_d))
    print(v, n, f)

    def draw_boxes(boxes):
        for b in boxes:
            b = b.cpu()
            mx, my, sx, sy = b[..., 0], b[..., 1], b[..., 3], b[..., 4]
            # xs = torch.cat([mx, mx + sx, mx + sx, mx, mx], dim=-1)
            # ys = torch.cat([my, my, my + sy, my + sy, my], dim=-1)
            xs = np.array([mx, mx + sx, mx + sx, mx, mx])
            ys = np.array([my, my, my + sy, my + sy, my])
            plt.plot(xs, ys)


    def tst_fn(boxes):
        centers = boxes[..., :2] + boxes[..., 3:5] * 0.5
        ofs = bin_to_oct(torch.linspace(0, 7, 8, dtype=torch.long, device=boxes.device).view(-1, 1), True)  # [8, 3]
        dist = (centers - 0.5).norm(dim=-1)
        return (dist - 0.4).abs() < 0.01

    # torch.manual_seed(1)
    #
    # # octree = Octree([0, 0, 0], [1, 1, 1])
    # # octree.build(tst_fn, 5, 3)
    # # x = torch.rand(8024, 3)
    # # # x = torch.tensor([[0.3, 0.7, 0.3]])
    # # ptr = octree.query(x)
    # # # draw_boxes(octree.boxes[ptr.flatten()])
    # # # plt.show()
    # #
    #
    octree = Octree([0, 0, 0], [1, 1, 1])
    octree.build(tst_fn, 10, 5)

    out = octree.query(torch.tensor([10, 10, 10], device="cuda"))
    print(out)
    #
    # st = time.time()
    # for i in range(1000):
    #     x = torch.rand(1024 * 256, 3).cuda()
    #     ptr = octree.query(x)
    # print(time.time() - st)
    #
    # res = tst_fn(octree.boxes[ptr]) * 100 + 1
    # stride = 1
    # plt.scatter(x[::stride, 0].cpu(), x[::stride, 1].cpu(), s=0.01, c=res[::stride].cpu())
    # plt.show()

    # stack = Stack(5, 8)
    # mask = torch.tensor([0, 1, 1, 0, 1], device="cuda").bool()
    # mask2 = torch.tensor([0, 0, 1, 0, 0], device="cuda").bool()
    # values = torch.tensor([3, 2, 4], device="cuda")
    # values2 = torch.tensor([1] * 5, device="cuda")
    # stack.push(values2)
    # stack.push(values, mask)
    # print(stack)
    # print(stack.peek())
    # stack.pop(mask2)
    # print(stack)
    # stack.pop()
    # print(stack)
    # print(stack.peek())

    for i in range(10000):
        x = torch.rand(50000, 3, device="cuda")
        a = octree.relative_query(x).peek()
        b = octree.query(x)
        if not (a == b[..., 0]).all():
            t = octree.query(x[(a != b[..., 0]).nonzero()[0]])
            print(t)
        print(".")
