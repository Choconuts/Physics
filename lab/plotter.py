
import matplotlib.pyplot as plt
import numpy as np

y = np.array([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 2.0355e-01, 3.2149e-01, 4.6643e-01, 0.0000e+00,
        0.0000e+00, 1.6724e-03, 4.1377e-03, 0.0000e+00, 5.4264e-04, 2.8085e-04,
        1.2963e-03, 0.0000e+00, 6.0403e-04, 0.0000e+00, 0.0000e+00, 5.7448e-14,
        2.8511e-15, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00])
z = [8.3333e-04, 3.9672e-03, 4.2634e-03, 1.2933e-02, 8.2155e-02, 3.2505e-01,
        2.3315e-02, 1.2087e-02, 1.2734e-02, 1.8905e-01, 2.8728e-01, 2.9981e-03,
        9.6147e-04, 8.8348e-04, 1.3522e-03, 1.1759e-03, 6.3247e-04, 1.6966e-02,
        6.0655e-03, 1.7618e-04, 2.6173e-04, 1.8747e-03, 2.2443e-03, 1.2117e-03,
        3.2419e-03, 1.2727e-03, 2.2577e-04, 2.8569e-04, 4.5059e-03, 2.5068e-14,
        1.4652e-14, 4.1087e-13, 9.1600e-24, 2.2436e-24, 1.2204e-24, 3.7958e-25,
        1.3592e-24, 2.2781e-25, 3.7915e-25, 5.6471e-25, 2.3760e-24, 1.6362e-24,
        3.8869e-25, 2.1640e-24, 2.1250e-24, 2.9148e-25, 4.3829e-25, 2.3343e-25,
        1.6566e-26, 1.0901e-24, 5.4129e-25, 3.5257e-25, 4.2057e-24, 4.1951e-25,
        2.1759e-24, 1.0402e-24, 1.2209e-26, 5.3663e-26, 2.9597e-25, 4.7939e-26,
        1.4560e-25, 1.4263e-25, 7.0827e-26, 6.9891e-26]
y = [0.1056, 0.1268, 0.1342, 0.1392, 0.1403, 0.1413, 0.1424, 0.1435, 0.1445,
        0.1456, 0.1467, 0.1477, 0.1488, 0.1499, 0.1502, 0.1505, 0.1508, 0.1510,
        0.1513, 0.1516, 0.1518, 0.1521, 0.1524, 0.1527, 0.1529, 0.1532, 0.1535,
        0.1537, 0.1540, 0.1543, 0.1545, 0.1548, 0.1551, 0.1553, 0.1556, 0.1559,
        0.1562, 0.1564, 0.1567, 0.1570, 0.1572, 0.1575, 0.1578, 0.1580, 0.1583,
        0.1586, 0.1588, 0.1591, 0.1594, 0.1597, 0.1599, 0.1602, 0.1605, 0.1607,
        0.1610, 0.1633, 0.1670, 0.1708, 0.1767, 0.1839, 0.1907, 0.1947, 0.1951,
        0.1956, 0.1960, 0.1965, 0.1970, 0.1974, 0.1979, 0.1984, 0.1988, 0.1993,
        0.1997, 0.2002, 0.2007, 0.2011, 0.2016, 0.2021, 0.2025, 0.2030, 0.2034,
        0.2039, 0.2044, 0.2048, 0.2053, 0.2057, 0.2060, 0.2063, 0.2066, 0.2069,
        0.2072, 0.2075, 0.2078, 0.2081, 0.2084, 0.2087, 0.2090, 0.2093, 0.2096,
        0.2100, 0.2103, 0.2106, 0.2109, 0.2112, 0.2115, 0.2118, 0.2121, 0.2124,
        0.2127, 0.2130, 0.2133, 0.2136, 0.2139, 0.2142, 0.2145, 0.2148, 0.2151,
        0.2154, 0.2157, 0.2160, 0.2163, 0.2167, 0.2735, 0.2881, 0.2933, 0.3055,
        0.3681, 0.7944]
x = np.linspace(2, 6, len(y))

# plt.plot(x, y)
# plt.show()

# x = np.linspace(-1, 1, 500)
# y = np.clip(1 - np.clip(x + 0.5, 0, 1), 0, 1)
# plt.figure(figsize=[4, 2])
# ax = plt.subplot(111)
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.spines['bottom'].set_position(('data', 0))
# ax.spines['left'].set_position(('data', 0))
# plt.xticks([-1, 1], ['-1', '1'])
# plt.yticks([0, 1], ['0', '1'])
# plt.plot(x, y, color='blue')
# plt.show()
# plt.savefig("activate.png")


z_vals = [0.1000, 0.1111, 0.1222, 0.1333, 0.1444, 0.1556, 0.1667, 0.1778, 0.1889,
        0.2000, 0.2111, 0.2222, 0.2333, 0.2444, 0.2556, 0.2667, 0.2778, 0.2889,
        0.3000, 0.3111, 0.3222, 0.3333, 0.3444, 0.3556, 0.3667, 0.3778, 0.3889,
        0.4000, 0.4111, 0.4222, 0.4333, 0.4444, 0.4556, 0.4667, 0.4778, 0.4889,
        0.5000, 0.5111, 0.5222, 0.5333, 0.5444, 0.5556, 0.5667, 0.5778, 0.5889,
        0.6000, 0.6111, 0.6222, 0.6333, 0.6444, 0.6556, 0.6667, 0.6778, 0.6889,
        0.7000, 0.7111, 0.7222, 0.7333, 0.7444, 0.7556, 0.7667, 0.7778, 0.7889,
        0.8000]
z_samples = [0.1056, 0.1268, 0.1342, 0.1392, 0.1403, 0.1413, 0.1424, 0.1435, 0.1445,
        0.1456, 0.1467, 0.1477, 0.1488, 0.1499, 0.1502, 0.1505, 0.1508, 0.1510,
        0.1513, 0.1516, 0.1518, 0.1521, 0.1524, 0.1527, 0.1529, 0.1532, 0.1535,
        0.1537, 0.1540, 0.1543, 0.1545, 0.1548, 0.1551, 0.1553, 0.1556, 0.1559,
        0.1562, 0.1564, 0.1567, 0.1570, 0.1572, 0.1575, 0.1578, 0.1580, 0.1583,
        0.1586, 0.1588, 0.1591, 0.1594, 0.1597, 0.1599, 0.1602, 0.1605, 0.1607,
        0.1610, 0.1633, 0.1670, 0.1708, 0.1767, 0.1839, 0.1907, 0.1947, 0.1951,
        0.1956, 0.1960, 0.1965, 0.1970, 0.1974, 0.1979, 0.1984, 0.1988, 0.1993,
        0.1997, 0.2002, 0.2007, 0.2011, 0.2016, 0.2021, 0.2025, 0.2030, 0.2034,
        0.2039, 0.2044, 0.2048, 0.2053, 0.2057, 0.2060, 0.2063, 0.2066, 0.2069,
        0.2072, 0.2075, 0.2078, 0.2081, 0.2084, 0.2087, 0.2090, 0.2093, 0.2096,
        0.2100, 0.2103, 0.2106, 0.2109, 0.2112, 0.2115, 0.2118, 0.2121, 0.2124,
        0.2127, 0.2130, 0.2133, 0.2136, 0.2139, 0.2142, 0.2145, 0.2148, 0.2151,
        0.2154, 0.2157, 0.2160, 0.2163, 0.2167, 0.2735, 0.2881, 0.2933, 0.3055,
        0.3681, 0.7944]
weights = [8.3333e-04, 3.9672e-03, 4.2634e-03, 1.2933e-02, 8.2155e-02, 3.2505e-01,
        2.3315e-02, 1.2087e-02, 1.2734e-02, 1.8905e-01, 2.8728e-01, 2.9981e-03,
        9.6147e-04, 8.8348e-04, 1.3522e-03, 1.1759e-03, 6.3247e-04, 1.6966e-02,
        6.0655e-03, 1.7618e-04, 2.6173e-04, 1.8747e-03, 2.2443e-03, 1.2117e-03,
        3.2419e-03, 1.2727e-03, 2.2577e-04, 2.8569e-04, 4.5059e-03, 2.5068e-14,
        1.4652e-14, 4.1087e-13, 9.1600e-24, 2.2436e-24, 1.2204e-24, 3.7958e-25,
        1.3592e-24, 2.2781e-25, 3.7915e-25, 5.6471e-25, 2.3760e-24, 1.6362e-24,
        3.8869e-25, 2.1640e-24, 2.1250e-24, 2.9148e-25, 4.3829e-25, 2.3343e-25,
        1.6566e-26, 1.0901e-24, 5.4129e-25, 3.5257e-25, 4.2057e-24, 4.1951e-25,
        2.1759e-24, 1.0402e-24, 1.2209e-26, 5.3663e-26, 2.9597e-25, 4.7939e-26,
        1.4560e-25, 1.4263e-25, 7.0827e-26, 6.9891e-26]


z_vals = np.array(z_vals)
z_samples = np.array(z_samples)
weights = np.array(weights)

all_z_vals = np.sort(np.concatenate([z_vals, z_samples]))


anchor = z_vals

def get_anchor_idx(z_vals, min_val, max_val, num_sample):
    anchor_idx = (z_vals - min_val) / (max_val - min_val)
    anchor_idx = anchor_idx * (num_sample - 1)
    anchor_idx = np.floor(anchor_idx)
    return anchor_idx

anchor_idx = get_anchor_idx(z_vals, z_vals[0], z_vals[-1], len(z_vals))

anchor_idx = np.floor((all_z_vals - z_vals[0]) / (z_vals[-1] - z_vals[0]) * (len(z_vals) - 1)).astype(np.int64)

ofs_anchor_idx = np.clip(anchor_idx - 1, 0, len(z_vals))

net_work = lambda x: (np.sign(x - 0.4) + 1) / 2
similarity = lambda x, y: np.abs(x - y)

anchor_theta = net_work(z_vals)
theta = net_work(all_z_vals)

anchor_alpha = similarity(anchor_theta[1:], anchor_theta[:-1])

anchor_T = np.cumprod(np.concatenate([np.ones(1), 1 - anchor_alpha]))
anchor_weight = anchor_T[:-1] * anchor_alpha

alpha = similarity(theta, anchor_theta[ofs_anchor_idx])

T = anchor_T[ofs_anchor_idx] * (1 - alpha)
last_alpha = 1 - T[1:] / np.where(T[:-1] > T[1:], T[:-1], T[1:])
weights = T[:-1] * last_alpha

print(anchor_idx)


x = np.linspace(0, 1, len(weights))
plt.plot(x, weights)
plt.show()



if __name__ == '__main__':
    pass