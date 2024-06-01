import matplotlib.pyplot as plt

# 功率
x_points1 = [-5, 0, 5, 10, 15]
y_proposed1 = [522,	528,	573,	630,	661

]
y_japb_c1 = [535,	519,	546,	596,	630


]
y_japb_s1 = [0.477,	0.505,	0.547,	0.560,	0.562]
y_RIS1 = [0.520, 0.525, 0.555, 0.587, 0.611]
y_jpba_r1 = [300, 300, 300, 300, 300, 300, 300]
y_cbbo1 = [195, 195, 195, 201, 201, 208, 227]

plt.figure(figsize=(8, 6))

plt.plot(x_points1, y_proposed1, label='proposed', marker='o')
plt.plot(x_points1, y_japb_c1, label='JAPB_C', marker='s')
# plt.plot(x_points1, y_japb_s1, label='JAPB_S', marker='d')
# plt.plot(x_points1, y_RIS1, label='RIS', marker='x')
# plt.plot(x_points1, y_jpba_r1, label='Joint Power and Bandwidth Allocation for Sensing', marker='*')
# plt.plot(x_points1, y_jtora1, label='Heuristic algorithm: Joint Task Offloading and Resource Allocation', marker='+')

plt.xlabel('Transmit Power Budget (dBm)')
plt.ylabel('Average Utility')
plt.title('Average Utility vs.Transmit Power Budget')
plt.legend()
plt.grid(True)
plt.show()

# 用户数
x_points2 = [2, 4, 6, 8, 10]
y_proposed2 = [0.399,	0.618,	0.695,	0.742,	0.769]
y_RIS2 = [0.387,	0.611,	0.677,	0.721,	0.752]
y_japb_s2 = [0.360,	0.562,	0.650,	0.694, 0.723]
y_japb_c2 = [0.359,	0.582,	0.670,	0.705,	0.728]
y_jpba_r2 = [325, 310, 260, 245, 252, 270]
y_cbbo2 = [195, 161, 141, 128, 128, 128]

plt.figure(figsize=(8, 6))

plt.plot(x_points2, y_proposed2, label='proposed', marker='o')
plt.plot(x_points2, y_japb_c2, label='JAPB_C', marker='s')
plt.plot(x_points2, y_japb_s2, label='JAPB_S', marker='d')
plt.plot(x_points2, y_RIS2, label='RIS', marker='x')
# plt.plot(x_points2, y_jpba_r2, label='Joint Power and Bandwidth Allocation for Sensing', marker='*')
# plt.plot(x_points2, y_jtora2, label='Heuristic algorithm: Joint Task Offloading and Resource Allocation', marker='+')

plt.xlabel('Number of legitimate users')
plt.ylabel('Average Utility')
plt.title('Average Utility vs. Number of legitimate users')
plt.legend()
plt.grid(True)
plt.show()
#
# N
x_points3 = [10 + i * 10 for i in range(5)]
y_proposed3 = [0.558, 0.583, 0.596, 0.622, 0.637]
y_RIS3 = [0.545,	0.558,	0.584,	0.599,	0.611]
y_japb_c3 = [0.538,	0.550,	0.572,	0.592,	0.606]
y_japb_s3 = [0.533, 0.545, 0.562, 0.577, 0.590]
y_jpba_r3 = [160, 325, 475, 650, 800, 970]
y_cbbo3 = [97, 195, 293, 393, 493, 593]

plt.figure(figsize=(8, 6))

plt.plot(x_points3, y_proposed3, label='proposed', marker='o')
plt.plot(x_points3, y_japb_c3, label='JAPB_C', marker='s')
plt.plot(x_points3, y_japb_s3, label='JAPB_S', marker='d')
plt.plot(x_points3, y_RIS3, label='RIS', marker='x')
# plt.plot(x_points3, y_jpba_r3, label='Joint Power and Bandwidth Allocation for Sensing', marker='*')
# plt.plot(x_points3, y_jtora3, label='Heuristic algorithm: Joint Task Offloading and Resource Allocation', marker='+')

plt.xlabel('Number of elements at the STAR-RIS')
plt.ylabel('Average Utility')
plt.title('Average Utility vs. Number of elements at the STAR-RIS')
plt.legend()
plt.grid(True)
plt.show()
#
# rad vs p
x_points4 = [-5, 0, 5, 10, 15]
y_proposed4 = [0.058,	0.176,	0.602,	1.97,	6.62]
y_japb_c4 = [0.049,	0.144,	0.46,	1.64,	4.89]
y_japb_s4 = [0.054,	0.162,	0.606,	1.65,	5.65]
y_jpba_c4 = [140, 190, 220, 250, 268, 280]
y_jpba_r4 = [296, 297, 305, 320, 325, 340]
y_cbbo4 = [119, 128, 162, 195, 229, 262]

plt.figure(figsize=(8, 6))

plt.plot(x_points4, y_proposed4, label='proposed', marker='o')
plt.plot(x_points4, y_japb_c4, label='JAPB_C', marker='s')
plt.plot(x_points4, y_japb_s4, label='JAPB_S', marker='d')
# plt.plot(x_points4, y_jpba_c4, label='Joint Power and Bandwidth Allocation for Communication', marker='x')
# plt.plot(x_points4, y_jpba_r4, label='Joint Power and Bandwidth Allocation for Sensing', marker='*')
# plt.plot(x_points4, y_jtora4, label='Heuristic algorithm: Joint Task Offloading and Resource Allocation', marker='+')

plt.xlabel('Transmit power budget (dBm)')
plt.ylabel('radar estimation rate (bps/Hz)')
plt.title('radar estimation rate (bps/Hz) vs. Transmit power budget (dBm)')
plt.legend()
plt.grid(True)
plt.show()
#
# sec vs p
x_points5 = [-5, 0, 5, 10, 15]
y_proposed5 = [1.020,	1.147,	1.212,	1.350,	1.443]
y_japb_c5 = [1.077,	1.241,	1.438,	1.478,	1.559]
y_japb_s5 = [0.863,	1.065,	1.128,	1.325,	1.327]
y_RIS5 = [982.9335645614436,
1016.6559396562636,
1209.0421868449384,
1374.6986767489677,
1421.1080595855667]
y_jpba_r5 = [325, 635, 895, 1227, 1600]
y_cbbo5 = [195, 397, 598, 799, 1010]

plt.figure(figsize=(8, 6))

plt.plot(x_points5, y_proposed5, label='proposed', marker='o')
plt.plot(x_points5, y_japb_c5, label='JAPB_C', marker='s')
plt.plot(x_points5, y_japb_s5, label='JAPB_S', marker='d')
# plt.plot(x_points1, y_RIS5, label='RIS', marker='x')
# plt.plot(x_points5, y_jpba_r5, label='Joint Power and Bandwidth Allocation for Sensing', marker='*')
# plt.plot(x_points5, y_jtora5, label='Heuristic algorithm: Joint Task Offloading and Resource Allocation', marker='+')

plt.xlabel('Transmit power budget (dBm)')
plt.ylabel('Secrecy Rate (bps/Hz)')
plt.title('Secrecy Rate vs. Transmit power budget')
plt.legend()
plt.grid(True)
plt.show()