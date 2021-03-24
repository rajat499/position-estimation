import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d
import math
n, m = 30, 30
T = 25
world = np.zeros([30, 30])

sensor_reach = np.asarray([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 
						   [0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5],
						   [0.5, 0.6, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6, 0.5],
						   [0.5, 0.6, 0.7, 0.8, 0.8, 0.8, 0.7, 0.6, 0.5],
						   [0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5],
						   [0.5, 0.6, 0.7, 0.8, 0.8, 0.8, 0.7, 0.6, 0.5],
						   [0.5, 0.6, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6, 0.5],
						   [0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5],
						   [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])

sensor_world = [[8, 15], [15, 15], [22, 15], [15, 22]]
sensors = [[15, 21], [15, 14], [15, 7], [22, 14]]
sensor_prob = [0.9, 0.8, 0.7, 0.6, 0.5]
num_sensors = len(sensors)

# four actions: up, right, down, left
motion_prob = np.asarray([0.4, 0.3, 0.1, 0.2])

l = [1,2,3]
l = l/np.sum(np.asarray(l))
a = np.sum(np.asarray(l))
motion_prob.shape

# translates the python matrix coordinates to world coordinates or vice-versa
def translate(x, y):
	x_new = n-1-x
	return y, x_new

# choose probabilistically - distribution 
def choose_prob(p_dist):
  p_dist = p_dist/np.sum(p_dist)
  r = np.random.rand()
  sum = p_dist[0]
  for i in range(p_dist.shape[0]-1):
    if r <= sum: return i
    else: sum += p_dist[i+1]
  return len(p_dist)-1

def update_coord(x, y, i):
  if i==0: x -= 1 #up
  elif i==1: y += 1 #right
  elif i==2: x += 1 # down
  elif i==3: y -= 1 # left
  return x, y

# index of direction of motion
def move_robot(loc):
  x, y = loc[0], loc[1]
  if x==0:
    if y==0: # top left corner
      d = choose_prob(motion_prob[1:3])+1
    elif y==29: # top right corner
      d = choose_prob(motion_prob[2:4])+2
    else: # top edge
      d = choose_prob(motion_prob[1:4])+1

  elif x==29:
    if y==0: # bottom left corner
      d = choose_prob(motion_prob[0:2])
    elif y==29: # bottom right corner
      d = choose_prob(np.asarray([motion_prob[0], 0, 0, motion_prob[3]]))
    else: # bottom edge
      d = choose_prob(np.asarray([motion_prob[0], motion_prob[1], 0, motion_prob[3]]))
  
  elif y==0: #left edge
    d = choose_prob(motion_prob[:3])
  
  elif y==29: #right edge
    d = choose_prob(np.asarray([motion_prob[0], 0, motion_prob[2], motion_prob[3]]))

  else:
    d = choose_prob(motion_prob)

  return update_coord(x, y, d)

# (a)
# assuming robot to start from [15, 15] - python coordinates
robot_loc = np.zeros([T+1, 2])
sensor_data = np.zeros([T+1, len(sensors)])

robot_loc[0, :] = [15, 15]
for t in range(T):
  robot_loc[t+1, :] = move_robot(robot_loc[t, :])
  # print(robot_loc[t+1])

for t in range(T+1):
  for i in range(len(sensors)):
    dist = int(np.max(np.abs(robot_loc[t, :] - np.asarray(sensors[i]))))
    if dist<5:
      p = sensor_prob[dist]
      sensor_data[t][i] = choose_prob([1-p, p]) # 0 -> not detected, 1 -> detected

# # Position predicted series plot

# fig, axs = plt.subplots(5,5, figsize=(15, 8), sharex=True, sharey=True)
# for i, ax in enumerate(axs.flat):
#     ax.scatter(robot_loc[i, 0]+0.5, robot_loc[i, 1]+0.5,s = 100,c = "r", marker = "o")
#     ax.set_title(f'Time Step : {i}')

# plt.xlim([int(min(predicted_loc[:, 0]))-2,int(max(predicted_loc[:, 0]))+2])
# plt.ylim([int(min(predicted_loc[:, 1]))-2,int(max(predicted_loc[:, 1]))+2])
# plt.xticks(range(int(min(predicted_loc[:, 0]))-2,int(max(predicted_loc[:, 0])+2)))
# plt.yticks(range(int(min(predicted_loc[:, 1]))-2,int(max(predicted_loc[:, 1])+2)))
# fig.text(0.5, 0.04, 'x coordinate', ha='center', va='center')
# fig.text(0.06, 0.5, 'y coordinate', ha='center', va='center', rotation='vertical')
# plt.grid(True)
# plt.rcParams["figure.figsize"] = (30,30)
# plt.show()


# (b) Filtering
Bel_x = []
bel = np.ones([n, m])/(n*m)

for t in range(T+1):
  if t==0:
    for x in range(n):
      for y in range(m):
        bel[x][y] = 1
        for s in range(num_sensors):
          dist = int(max(abs(np.asarray([x-sensors[s][0], y-sensors[s][1]]))))
          p = sensor_prob[dist] if dist<5 else 0
          bel[x][y] *= p if sensor_data[t, s]==1 else (1-p)
    Bel_x.append(bel/np.sum(bel))
  else:
    for x in range(n):
      for y in range(m):
        tmp = 0
        if x>0: #from up
          tmp += motion_prob[2]*Bel_x[t-1][x-1][y]
        if y<29: #from right
          tmp += motion_prob[3]*Bel_x[t-1][x][y+1]
        if x<29: #from down
          tmp += motion_prob[1]*Bel_x[t-1][x+1][y]
        if y>0: #from left
          tmp += motion_prob[1]*Bel_x[t-1][x][y-1]
        for s in range(num_sensors):
          dist = int(max(abs(np.asarray([x-sensors[s][0], y-sensors[s][1]]))))
          p = sensor_prob[dist] if dist<5 else 0
          tmp *= p if sensor_data[t, s]==1 else (1-p)
        bel[x][y] = tmp
    Bel_x.append(bel/np.sum(bel))

predicted_path_b = np.zeros([T+1, 2])
for t in range(T+1):
  predicted_path_b[t, :] = np.unravel_index(np.argmax(Bel_x[t], axis=None), Bel_x[t].shape)

for t in range(T+1):
  loc = np.unravel_index(np.argmax(Bel_x[t], axis=None), Bel_x[t].shape)
  predicted_loc[t,:] = loc

# Position predicted series plot - b

# plt.rcParams['axes.facecolor'] = 'black'
# fig, axs = plt.subplots(5,5, figsize=(15, 8), sharex=True, sharey=True)
# for i, ax in enumerate(axs.flat):
#     ax.scatter(predicted_loc[i-1, 0]+0.5, predicted_loc[i-1, 1]+0.5,s = 100,c = "white", marker = "s")
#     ax.scatter(robot_loc[i-1, 0]+0.5, robot_loc[i-1, 1]+0.5,s = 100,c = "r", marker = "o")
#     ax.set_title(f'Time Step : {i}')

# plt.xlim([int(min(predicted_loc[:, 0]))-2,int(max(predicted_loc[:, 0]))+2])
# plt.ylim([int(min(predicted_loc[:, 1]))-2,int(max(predicted_loc[:, 1]))+2])
# plt.xticks(range(int(min(predicted_loc[:, 0]))-2,int(max(predicted_loc[:, 0])+2)))
# plt.yticks(range(int(min(predicted_loc[:, 1]))-2,int(max(predicted_loc[:, 1])+2)))
# fig.text(0.5, 0.04, 'x coordinate', ha='center', va='center')
# fig.text(0.06, 0.5, 'y coordinate', ha='center', va='center', rotation='vertical')
# plt.grid(True)
# plt.rcParams["figure.figsize"] = (30,30)
# plt.show()


# HEAT MAP Plotting
# Bel_x = np.asarray(Bel_x )
# Bel_x[Bel_x<10e-20] = 10e-20

# # Here we create a figure instance, and two subplots
# fig, axs = plt.subplots(5,5, figsize=(15, 8), sharex=True, sharey=True)
# # ax1 = fig.add_subplot(3, 3, 1) # row, column, position

# for i, ax in enumerate(axs.flat):
#     # ax.scatter(predicted_loc[i, 0]+0.5, predicted_loc[i, 1]+0.5,s = 100,c = "white", marker = "s")
#     ax.set_title(f'Time Step : {i}')
#     sns.heatmap(data=Bel_x[i], ax=ax,  square=True)


# LOG PLOTS Plotting
# x = y = np.arange(0, 30, 1)
# X, Y = np.meshgrid(x, y)
# fig = plt.figure(figsize=(30,30))
# for t in range(1,T+1):
#     Z = np.log(Bel_x[t])
#     ax = fig.add_subplot(5, 5, t, projection='3d')
#     ax.set_zlabel("Log Likelihood", labelpad=10)
#     # plt.zlabel("Log Likelihood",labelpad=10)
#     ax.set_xlabel("X coordinate")
#     ax.set_ylabel("Y coordinate")
#     ax.text2D(0.5, 0.95, "Time Step: " + str(t), transform=ax.transAxes)
#     p = ax.plot_surface(X,Y,Z,rstride=1, cstride=1,cmap='summer', edgecolor='none')

def sensor_model(sensor_data, t, x, y):
  tmp = 1
  for s in range(num_sensors):
    dist = int(max(abs(np.asarray([x-sensors[s][0], y-sensors[s][1]]))))
    p = sensor_prob[dist] if dist<5 else 0
    tmp *= p if sensor_data[t, s]==1 else (1-p)
  return tmp

# (c) Smoothing
smooth = np.zeros([T+1, n, m])
smooth[T] = Bel_x[T]

for t in reversed(range(T)):
  for x in range(n):
    for y in range(m):
      bel[x][y] = Bel_x[t][x][y]
      tmp = 0
      if x>0: #to up
        s_tmp = sensor_model(sensor_data, t+1, x-1, y)
        s_tmp *= (smooth[t+1][x-1][y]*motion_prob[0])
        tmp += s_tmp
      if y<29: #to right
        s_tmp = sensor_model(sensor_data, t+1, x, y+1)
        s_tmp *= (smooth[t+1][x][y+1]*motion_prob[1])
        tmp += s_tmp
      if x<29: #to down
        s_tmp = sensor_model(sensor_data, t+1, x+1, y)
        s_tmp *= (smooth[t+1][x+1][y]*motion_prob[2])
        tmp += s_tmp
      if y>0: #to left
        s_tmp = sensor_model(sensor_data, t+1, x, y-1)
        s_tmp *= (smooth[t+1][x][y-1]*motion_prob[3])
        tmp += s_tmp
      bel[x][y] *= tmp
  smooth[t] = bel/np.sum(bel)

predicted_path_c = np.zeros([T+1, 2])
for t in range(T+1):
  predicted_path_c[t, :] = np.unravel_index(np.argmax(smooth[t], axis=None), smooth[t].shape)

# # Position predicted series plot - C
# plt.rcParams['axes.facecolor'] = 'black'
# fig, axs = plt.subplots(5,5, figsize=(15, 8), sharex=True, sharey=True)
# for i, ax in enumerate(axs.flat):
#     print(robot_loc[i-1])
#     ax.scatter(predicted_path_c[i-1, 0]+0.5, predicted_path_c[i-1, 1]+0.5,s = 100,c = "white", marker = "s")
#     ax.scatter(robot_loc[i-1, 0]+0.5, robot_loc[i-1, 1]+0.5,s = 100,c = "r", marker = "o")
#     ax.set_title(f'Time Step : {i}')

# plt.xlim([int(min(predicted_path_c[:, 0]))-2,int(max(predicted_path_c[:, 0]))+2])
# plt.ylim([int(min(predicted_path_c[:, 1]))-2,int(max(predicted_path_c[:, 1]))+2])
# plt.xticks(range(int(min(predicted_path_c[:, 0]))-2,int(max(predicted_path_c[:, 0])+2)))
# plt.yticks(range(int(min(predicted_path_c[:, 1]))-2,int(max(predicted_path_c[:, 1])+2)))
# # set labels
# # plt.setp(axs[-1, :], xlabel='x')
# # plt.setp(axs[:, 0], ylabel='y')
# fig.text(0.5, 0.04, 'x coordinate', ha='center', va='center')
# fig.text(0.06, 0.5, 'y coordinate', ha='center', va='center', rotation='vertical')
# plt.grid(True)
# plt.rcParams["figure.figsize"] = (30,30)
# plt.show()


# (d)
import matplotlib.pyplot as plt
x_series = np.linspace(0, T, T+1)
man_error_b = (np.max(np.abs(predicted_path_b-robot_loc), axis=1))
man_error_c = (np.max(np.abs(predicted_path_c-robot_loc), axis=1))

fig, ax = plt.subplots(1, 1, figsize =(10, 7))
ax.plot(x_series, man_error_c, color="navy")
ax.set_title("Manhattan Distance - part c")
ax.set_xlabel("Time Step")
ax.set_ylabel("Manhattan Distance")
ax.legend(loc='lower right')
plt.show()

# (e) Future Prediction : set T_fut = 10, T_fut = 25
Bel_future = [Bel_x[T]]
T_fut = 25
bel = np.ones([n, m])/(n*m)
for t in range(1,T_fut+1):
  for x in range(n):
    for y in range(m):
      tmp = 0
      if x>0: #from up
        tmp += motion_prob[2]*Bel_future[t-1][x-1][y]
      if y<29: #from right
        tmp += motion_prob[3]*Bel_future[t-1][x][y+1]
      if x<29: #from down
        tmp += motion_prob[1]*Bel_future[t-1][x+1][y]
      if y>0: #from left
        tmp += motion_prob[1]*Bel_future[t-1][x][y-1]
      bel[x][y] = tmp
  Bel_future.append(bel/np.sum(bel))

Bel_future = np.asarray(Bel_future )
Bel_future[Bel_future<10e-10] = 10e-10


# LOG PLOT Future prediction
# x = y = np.arange(0, 30, 1)
# X, Y = np.meshgrid(x, y)
# fig = plt.figure(figsize=(30,30))
# for t in range(1,26):
#     Z = Bel_future[t]
#     ax = fig.add_subplot(5, 5, t, projection='3d')
#     ax.set_zlabel("Log Likelihood", labelpad=10)
#     # plt.zlabel("Log Likelihood",labelpad=10)
#     ax.set_xlabel("X coordinate")
#     ax.set_ylabel("Y coordinate")
#     ax.text2D(0.5, 0.95, "Time Step: " + str(t), transform=ax.transAxes)
#     p = ax.plot_surface(X,Y,Z,rstride=1, cstride=1,cmap='summer', edgecolor='none')

# # To print belief at time t
# l = []
# t = 10
# for x in range(n):
#   for y in range(m):
#     print(round(Bel_future[t][x][y], 3), end="  ")
#     if Bel_future[t][x][y]>0.0:
#       l.append([x,y])
#   print("\n")

# print(np.unravel_index(np.argmax(Bel_future[t], axis=None), Bel_future[t].shape))
# print(robot_loc[t])

# (f) Most Likely Path
paths = []
MLE = []
Most_likely = np.zeros([n,m])
for x in range(n):
  for y in range(m):
    paths.append([(x,y)])
    Most_likely[x][y] = sensor_model(sensor_data, 0, x, y)
Most_likely = Most_likely/np.sum(Most_likely)
MLE.append(Most_likely)

for t in range(1, T+1):
  paths_old = paths
  for x in range(n):
    for y in range(m):
      tmp = []
      xt = []
      if x>0: #from up
        tmp.append(motion_prob[2]*MLE[t-1][x-1][y])
        xt.append([x-1,y])
      if y<29: #from right
        tmp.append(motion_prob[3]*MLE[t-1][x][y+1])
        xt.append([x,y+1])
      if x<29: #from down
        tmp.append(motion_prob[1]*MLE[t-1][x+1][y])
        xt.append([x+1,y])
      if y>0: #from left
        tmp.append(motion_prob[1]*MLE[t-1][x][y-1])
        xt.append([x,y-1])

      ind = np.argmax(tmp)
      paths[(x*n)+y] = paths_old[(xt[ind][0]*n) + xt[ind][1]] + [(x, y)]
      Most_likely[x][y] = np.max(tmp)*sensor_model(sensor_data, t, x, y)
  
  MLE.append(Most_likely/np.sum(Most_likely))

predicted_MLE = np.zeros([T+1, 2])
for t in range(T+1):
  predicted_MLE[t, :] = np.unravel_index(np.argmax(predicted_MLE[t], axis=None), predicted_MLE[t].shape)


# Position predicted series plot - F
# plt.rcParams['axes.facecolor'] = 'black'
# fig, axs = plt.subplots(5,5, figsize=(15, 8), sharex=True, sharey=True)
# for i, ax in enumerate(axs.flat):
#     ax.scatter(predicted_MLE[i-1, 0]+0.5, predicted_MLE[i-1, 1]+0.5,s = 100,c = "white", marker = "s")
#     ax.scatter(robot_loc[i-1, 0]+0.5, robot_loc[i-1, 1]+0.5,s = 100,c = "r", marker = "o")
#     ax.set_title(f'Time Step : {i}')

# plt.xlim([int(min(predicted_MLE[:, 0]))-2,int(max(predicted_MLE[:, 0]))+2])
# plt.ylim([int(min(predicted_MLE[:, 1]))-2,int(max(predicted_MLE[:, 1]))+2])
# plt.xticks(range(int(min(predicted_MLE[:, 0]))-2,int(max(predicted_MLE[:, 0])+2)))
# plt.yticks(range(int(min(predicted_MLE[:, 1]))-2,int(max(predicted_MLE[:, 1])+2)))
# # set labels
# # plt.setp(axs[-1, :], xlabel='x')
# # plt.setp(axs[:, 0], ylabel='y')
# fig.text(0.5, 0.04, 'x coordinate', ha='center', va='center')
# fig.text(0.06, 0.5, 'y coordinate', ha='center', va='center', rotation='vertical')
# plt.grid(True)
# plt.rcParams["figure.figsize"] = (30,30)
# plt.show()

