import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


## Parameters
L1 = 0.4
L2 = 0.5
L3 = 0.3

## IK for 3R arm
def inverse_kinematics(x, y, z):
    # Solve for q1
    q1 = np.arctan2(y, x)

    # Distance in horizontal plane
    D = np.sqrt(x**2 + y**2)

    # Distance from p1 to p3 
    D2 = D**2 + (z - L1)**2
    r = np.sqrt(D2)

    # Solve for q3
    cos_q3 = (D2 - L2**2 - L3**2) / (2 * L2 * L3) # Law of cosine
    cos_q3 = np.clip(cos_q3, -1, 1) # avoids errors
    q3 = -np.arccos(cos_q3)

    # Solve for q2
    k1 = L2 + L3 * np.cos(q3)
    k2 = L3 * np.sin(q3)
    gamma = np.arctan2((z - L1), D)
    alpha = np.arctan2(k2, k1)
    q2 = gamma - alpha

    return np.array([q1, q2, q3])


## Homogenous rotation matrices
def Rz(theta):
    cq = np.cos(theta)
    sq = np.sin(theta)
    return np.array([
        [ cq, -sq, 0],
        [ sq,  cq, 0],
        [0, 0, 1]
    ])

def Ry(theta):
    cq = np.cos(theta)
    sq = np.sin(theta)
    return np.array([
        [ cq, 0,  sq],
        [0, 1, 0],
        [-sq, 0,  cq]
    ])


## FK for 3R arm
def forward_kinematics(q):
    # Uses homogenous transformation matrices to compute FK

    q1 = q[0]
    q2 = q[1]
    q3 = q[2]

    # Rotations
    R01 = Rz(q1)
    R12 = Ry(-q2)
    R23 = Ry(-q3)

    # Total rotation
    R02 = R01 @ R12
    R03 = R02 @ R23

    # Finds end-effector position
    p1 = R01 @ np.array([0, 0, L1])
    p2 = p1 + R02 @ np.array([L2, 0, 0])
    p3 = p2 + R03 @ np.array([L3, 0, 0])

    return p3


## Pick & place points
pick_xyz  = np.array([0.5,  0.5, 0])
place_xyz = np.array([-0.5, -0.1, 0])

# End effector position at start
home_xyz  = np.array([0.6,  0, 0.3])

# Move betweem points with IK
home = inverse_kinematics(*home_xyz)
pick = inverse_kinematics(*pick_xyz)
place = inverse_kinematics(*place_xyz)


## Timing
T1  = 3   # home to pick
Th1 = 1   # hold at pick
T2  = 3   # pick to place
Th2 = 1   # hold at place

T_total = T1 + Th1 + T2 + Th2


## Simulation parameters
dt = 0.01
N = int(T_total / dt) + 1
time = np.linspace(0, T_total, N)

## Controlled movement functions
def velocity_control(q0, qf, T, f):
    # Used cubic polynomial with zero velocity at start and end - found this method online
    # Returns q_d, dq_d

    # setup
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

# PD controller
# f = Kp*(q_d - q) + Kd*(dq_d - dq)
Kp = np.array([50, 50, 50])
Kd = np.array([10, 10, 10])

# Time stepping
for i in range(N - 1):
    t = time[i]
    qd[i, :], dqd[i, :] = desired_trajectory(t)

    # PD control
    e  = qd[i, :] - q[i, :] # position error
    de = dqd[i, :] - dq[i, :] # velocity error
    f  = Kp * e + Kd * de

    ddq = f

    # integration
    dq[i + 1, :] = dq[i, :] + ddq * dt
    q[i + 1, :]  = q[i, :] + dq[i + 1, :] * dt

# Compute at final step
qd[-1, :], dqd[-1, :] = desired_trajectory(time[-1])


## End-effector trajectories
x  = np.zeros(N)
y  = np.zeros(N)
z  = np.zeros(N)
xd = np.zeros(N)
yd = np.zeros(N)
zd = np.zeros(N)

# end effector positions
for i in range(N):
    x[i],  y[i],  z[i] = forward_kinematics(q[i, :])
    xd[i], yd[i], zd[i] = forward_kinematics(qd[i, :])

# Pick and place points
pick_xyz_fk = forward_kinematics(pick)
place_xyz_fk = forward_kinematics(place)

# Final error
final_error = np.linalg.norm(forward_kinematics(q[-1, :]) - place_xyz_fk)
print(f"Final end-effector error: {final_error:.4f}")


## Plots
plt.figure(figsize=(12, 4))

# x(t)
plt.subplot(1, 3, 1)
plt.plot(time, x, label='x actual')
plt.plot(time, xd, '--', label='x desired')
plt.axhline(pick_xyz_fk[0], linestyle=':', label='Pick x')
plt.axhline(place_xyz_fk[0], linestyle=':', label='Place x')
plt.xlabel('Time')
plt.ylabel('x')
plt.title('End-Effector X Trajectory')
plt.legend()

# y(t)
plt.subplot(1, 3, 2)
plt.plot(time, y, label='y actual')
plt.plot(time, yd, '--', label='y desired')
plt.axhline(pick_xyz_fk[1], linestyle=':', label='Pick y')
plt.axhline(place_xyz_fk[1], linestyle=':', label='Place y')
plt.xlabel('Time')
plt.ylabel('y')
plt.title('End-Effector Y Trajectory')
plt.legend()
 # z(t)
plt.subplot(1, 3, 3)
plt.plot(time, z, label='z actual')
plt.plot(time, zd, '--', label='z desired')
plt.axhline(0.0, linestyle=':', label='Ground')
plt.xlabel('Time')
plt.ylabel('z')
plt.title('End-Effector Z Trajectory')
plt.legend()

# prevents overlap
plt.tight_layout()


## Animation
# Link positions
x0 = np.zeros(N)
y0 = np.zeros(N)
z0 = np.zeros(N)

x1 = np.zeros(N)
y1 = np.zeros(N)
z1 = np.full(N, L1)

x2 = np.zeros(N)
y2 = np.zeros(N)
z2 = np.zeros(N)

x3 = np.zeros(N)
y3 = np.zeros(N)
z3 = np.zeros(N)

# unit vector in z direction
dz = np.array([0, 0, 1])

# Compute link positions
for i in range(N):
    q1, q2, q3 = q[i, :]
    dr_i = np.array([np.cos(q1), np.sin(q1), 0])

    # Base and first joint
    p0 = np.array([0, 0, 0])
    p1 = np.array([0, 0, L1])

    # Rotation matrices
    d2 = np.cos(q2) * dr_i + np.sin(q2) * dz
    d3 = np.cos(q2 + q3) * dr_i + np.sin(q2 + q3) * dz

    # Link positions
    p2 = p1 + L2 * d2
    p3 = p2 + L3 * d3

    x0[i], y0[i], z0[i] = p0
    x1[i], y1[i], z1[i] = p1
    x2[i], y2[i], z2[i] = p2
    x3[i], y3[i], z3[i] = p3

# figure setup
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3-DoF Pick-and-Place")

# set limits
max_reach = L1 + L2 + L3 + 0.1
ax.set_xlim(-max_reach, max_reach)
ax.set_ylim(-max_reach, max_reach)
ax.set_zlim(0, max_reach)

# plot pick and place points
ax.scatter(pick_xyz[0],  pick_xyz[1],  pick_xyz[2], label='Pick')
ax.scatter(place_xyz[0], place_xyz[1], place_xyz[2], label='Place')

# robot links and joints
link_line, = ax.plot([], [], [], '-o', lw=3)
ax.legend(loc='upper right')

## Animation functions
# Initialize function
def init_3d():
    link_line.set_data_3d([], [], [])
    return link_line,

# Update function
def update_3d(frame):
    # frame index
    i = frame
    xs = [x0[i], x1[i], x2[i], x3[i]]
    ys = [y0[i], y1[i], y2[i], y3[i]]
    zs = [z0[i], z1[i], z2[i], z3[i]]
    link_line.set_data_3d(xs, ys, zs)
    return link_line,

# Animation speed
step = 5
frames = range(0, N, step)
interval_ms = dt * step * 1000

# Create animation
ani = FuncAnimation(fig, update_3d, frames=frames,
                    init_func=init_3d, blit=True,
                    interval=interval_ms)

plt.show()
