import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


## Parameters
L1 = 0.5
L2 = 0.3
L3 = 0.2  # third link

# Joint limits (not enforced, just kept)
joint_min = -90
joint_max = 90

# Pick and place setup (3-DoF)
home_deg  = np.array([  0.0,   0.0,   0.0])
pick_deg  = np.array([ 30.0, -20.0,  10.0])
place_deg = np.array([ 45.0, -30.0, -15.0])

# Convert to radians
home  = np.deg2rad(home_deg)
pick  = np.deg2rad(pick_deg)
place = np.deg2rad(place_deg)


## PD controller
# f = Kp*(q_d - q) + Kd*(dq_d - dq)
Kp = np.array([50.0, 50.0, 50.0])
Kd = np.array([10.0, 10.0, 10.0])


## Trajectory parameters
T1  = 2.0   # home to pick
Th1 = 1.0   # hold at pick
T2  = 2.0   # pick to place
Th2 = 1.0   # hold at place

T_total = T1 + Th1 + T2 + Th2


## Simulation parameters
dt = 0.01 
N = int(T_total / dt) + 1 
time = np.linspace(0, T_total, N) 


## Functions
def forward_kinematics(q):
    # Returns end effector position for joint angles (3R planar)
    q1, q2, q3 = q
    x1 = L1 * np.cos(q1)
    y1 = L1 * np.sin(q1)
    x2 = x1 + L2 * np.cos(q1 + q2)
    y2 = y1 + L2 * np.sin(q1 + q2)
    x3 = x2 + L3 * np.cos(q1 + q2 + q3)
    y3 = y2 + L3 * np.sin(q1 + q2 + q3)

    return np.array([x3, y3])

def velocity_control(q0, qf, T, f):
    # Cubic polynomial with zero velocity at start and end
    s = f / T
    s2 = s * s
    s3 = s2 * s

    # Position
    qd = q0 + (3 * s2 - 2 * s3) * (qf - q0)
    
    # Velocity
    dqd = (6 * s - 6 * s2) * (qf - q0) / T

    return qd, dqd

def desired_trajectory(t):
    # Time schedule for the whole pick-and-place task

    if t < 0:
        return home, np.zeros(3)

    if t < T1:
        f = t
        return velocity_control(home, pick, T1, f)

    if t < T1 + Th1:
        return pick, np.zeros(3)

    if t < T1 + Th1 + T2:
        f = t - (T1 + Th1)
        return velocity_control(pick, place, T2, f)

    return place, np.zeros(3)


## Simulation
q   = np.zeros((N, 3))
dq  = np.zeros((N, 3))
qd  = np.zeros((N, 3))
dqd = np.zeros((N, 3))

# Start at home
q[0, :] = home
dq[0, :] = 0.0

# Time step
for i in range(N - 1):
    t = time[i]
    qd[i, :], dqd[i, :] = desired_trajectory(t)

    # PD control
    e  = qd[i, :] - q[i, :]   # position error
    de = dqd[i, :] - dq[i, :] # velocity error
    f  = Kp * e + Kd * de

    # No gravity for simplicity
    ddq = f

    # Integrate 
    dq[i + 1, :] = dq[i, :] + ddq * dt
    q[i + 1, :]  = q[i, :] + dq[i + 1, :] * dt

# Compute at final step
qd[-1, :], dqd[-1, :] = desired_trajectory(time[-1])

# End-effector trajectories
x  = np.zeros(N)
y  = np.zeros(N)
xd = np.zeros(N)
yd = np.zeros(N)

# end effector positions
for i in range(N):
    x[i],  y[i]  = forward_kinematics(q[i, :])
    xd[i], yd[i] = forward_kinematics(qd[i, :])

# Pick and Place points (in task space)
pick_xy  = forward_kinematics(pick)
place_xy = forward_kinematics(place)

# Final error
final_error = np.linalg.norm(forward_kinematics(q[-1, :]) - place_xy)
print(f"Final end-effector error: {final_error:.4f}")


## Plots
plt.figure(figsize=(8, 4))

# x(t)
plt.subplot(1, 2, 1)
plt.plot(time, x, label='x actual')
plt.plot(time, xd, '--', label='x desired')
plt.axhline(pick_xy[0], linestyle=':', label='Pick x')
plt.axhline(place_xy[0], linestyle=':', label='Place x')
plt.xlabel('Time')
plt.ylabel('x')
plt.title('End-Effector X Trajectory')
plt.legend()

# y(t)
plt.subplot(1, 2, 2)
plt.plot(time, y, label='y actual')
plt.plot(time, yd, '--', label='y desired')
plt.axhline(pick_xy[1], linestyle=':', label='Pick y')
plt.axhline(place_xy[1], linestyle=':', label='Place y')
plt.xlabel('Time')
plt.ylabel('y')
plt.title('End-Effector Y Trajectory')
plt.legend()

plt.tight_layout()


## Animation
# Joint positions over time
x0 = np.zeros(N)
y0 = np.zeros(N)

# first joint
x1 = L1 * np.cos(q[:, 0])
y1 = L1 * np.sin(q[:, 0])

# second joint
x2 = x1 + L2 * np.cos(q[:, 0] + q[:, 1])
y2 = y1 + L2 * np.sin(q[:, 0] + q[:, 1])

# end-effector (third joint)
x3 = x
y3 = y

# figure setup
fig, ax = plt.subplots()
ax.set_aspect('equal', 'box')

# set limits
max_reach = L1 + L2 + L3 + 0.1
ax.set_xlim(-max_reach, max_reach)
ax.set_ylim(-max_reach, max_reach)
ax.set_title("3-Link Pick-and-Place")

# Plot points for pick and place 
ax.scatter(pick_xy[0], pick_xy[1], label='Pick')
ax.scatter(place_xy[0], place_xy[1], label='Place')

# robot links and joints
link_line, = ax.plot([], [], '-o', lw=3)
ax.legend(loc='upper right')

## Animation functions
def init():
    link_line.set_data([], [])
    return link_line,

def update(frame):
    i = frame
    xs = [x0[i], x1[i], x2[i], x3[i]]
    ys = [y0[i], y1[i], y2[i], y3[i]]
    link_line.set_data(xs, ys)
    return link_line,

# Animation speed 
step = 5
frames = range(0, N, step)
interval_ms = dt * step * 1000

ani = FuncAnimation(fig, update, frames=frames, init_func=init,
                    blit=True, interval=interval_ms)

plt.show()
