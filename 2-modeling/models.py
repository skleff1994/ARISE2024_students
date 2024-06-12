import numpy as np
from matplotlib import pyplot as plt
import IPython
from IPython.display import Image
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


# Class for the geometric model of the 2R robot
class GeometricModel1R:
    '''
    Environment class for the geometric model of the 1R robot
    '''
    
    def __init__(self, L=1.):
        ''' 
        Initialize model parameters
        '''
        self.L = L
    
    def DGM(self, q):
        '''
        Input : joint angle (in rad)
        Output : end-effector position (in m)
        '''
        x_p = self.L*np.cos(q)
        y_p = self.L*np.sin(q)
        p = np.array([x_p, y_p])
        return p

    def IGM(self, p):
        '''
        Input : end-effector position (in m)
        Output : joint angle (in rad)
        '''
        x_p = p[0]
        y_p = p[1]
        q = np.arctan2(y_p, x_p) % (2*np.pi)
        return q
    
    def animate(self, q):
        '''
        Simulate the robot using DGM 
        '''
        fig = plt.figure()
        ax = plt.axes(xlim=(-self.L -1, self.L + 1), ylim=(-self.L -1, self.L + 1))
        text_str = "One Dof Manipulator Animation"
        link, = ax.plot([], [], lw=4)
        base, = ax.plot([], [], 'o', color='black')
        endeff, = ax.plot([], [], 'o', color='pink')
        
        def init():
            link.set_data([], [])
            base.set_data([], [])
            endeff.set_data([], [])
            return link, base, endeff
        
        def animate(i):
            p = self.DGM(q[i])
            x = p[0] 
            y = p[1] 
            link.set_data([0,x], [0,y])
            base.set_data([0, 0])
            endeff.set_data([x, y])
            return link, base, endeff
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=15,
        verticalalignment='top', bbox=props)
        ax.grid()
        anim = FuncAnimation(fig, animate, init_func=init,
                                       frames=np.shape(q)[0], interval=50, blit=True)

        plt.close(fig)
        plt.close(anim._fig)
        IPython.display.display_html(IPython.core.display.HTML(anim.to_html5_video()))


class KinematicModel1R:
    '''
    Environment class for the 1R robot kinematics
    '''
    def __init__(self, L=1.):
        ''' 
        Initialize model parameters
        '''
        self.L = L
    
    def DKM(self, q, qdot):
        '''
        Direct Kinematic Model function
        Input : joint angular position and velocity (in rad and rad/s)
        Output : end-effector position (in m/s)
        '''
        xdot_p = -self.L*np.sin(q)*qdot
        ydot_p = self.L*np.cos(q)*qdot
        pdot = np.array([xdot_p, ydot_p])
        return pdot

    def IKM(self, q, pdot):
        '''
        Inverse Kinematic Model function
        Input : end-effector position (in m), joint position (m)
        Output : joint velocity (in rad/s)
        '''    
        xdot_p = pdot[0]
        ydot_p = pdot[1]
        qdot = (ydot_p*np.cos(q) - xdot_p*np.sin(q)) / self.L
        return qdot


class GeometricModel2R:
    '''
    Environment class for the geometric model of the 1R robot
    '''
    
    def __init__(self, L1=1., L2=1.):
        ''' 
        Initialize model parameters
        '''
        
        self.L1 = L1
        self.L2 = L2
        
    def DGM(self, q):
        '''
        Input : joint positions q=(q1,q2) (in rad)
        Output : end-effector position (in m)
        '''
        q1 = q[0]
        q2 = q[1]
        x_p = self.L1*np.cos(q1) + self.L2*np.cos(q1 + q2)
        y_p = self.L1*np.sin(q1) + self.L2*np.sin(q1 + q2)
        p = np.array([x_p, y_p])
        return p

    def IGM(self, p):
        '''
        Input : end-effector position (in m)
        Output : joint positions q (in rad)
        '''
        x_p = p[0]
        y_p = p[1]
        q2 = np.arccos( (x_p**2 + y_p**2 - self.L1**2 - self.L2**2) / (2*self.L1*self.L2) )
        q1 = ( np.arctan2(y_p, x_p) - q2 )# % (2*np.pi)
        q = np.array([q1, q2]) # % (2*np.pi)
        return q
        
    def animate(self, q):
        '''
        Simulate the robot using DGM 
        Input : sequence of joint positions [q1, ..., q2]
        '''
        fig = plt.figure()
        ax = plt.axes(xlim=(-self.L1 -1, self.L1 + 1), ylim=(-self.L2 -1, self.L2 + 1))
        text_str = "2R robot animation"
        link1, = ax.plot([], [], lw=4)
        link2, = ax.plot([], [], lw=4)
        base, = ax.plot([], [], 'o', color='black')
        endeff, = ax.plot([], [], 'o', color='pink')
        
        def init():
            link1.set_data([], [])
            link2.set_data([], [])
            base.set_data([], [])
            endeff.set_data([], [])
            return link1, link2, base, endeff
        
        def animate(i):
            p = self.DGM(q[:,i])
            x = p[0] 
            y = p[1] 
            link1.set_data([0,self.L1*np.cos(q[0,i])], [0,self.L1*np.sin(q[0,i])])
            link2.set_data([self.L1*np.cos(q[0,i]),x], [self.L1*np.sin(q[0,i]),y])
            base.set_data([0, 0])
            endeff.set_data([x, y])
            return link1, link2, base, endeff
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=15,
        verticalalignment='top', bbox=props)
        ax.grid()
        anim = FuncAnimation(fig, animate, init_func=init,
                                       frames=np.shape(q)[1], interval=25, blit=True)

        plt.close(fig)
        plt.close(anim._fig)
        IPython.display.display_html(IPython.core.display.HTML(anim.to_html5_video()))


class KinematicModel2R:
    '''
    Environment class for the 2R robot kinematics
    '''
    def __init__(self, L1=1., L2=1.):
        ''' 
        Initialize model parameters
        '''
        self.L1 = L1
        self.L2 = L2

    def jacobian(self, q):
        '''
        Calculate the 2x2 Jacobian matrix 
        Input : joint position q
        Output : Jacobian matrix
        '''
        q1 = q[0]
        q2 = q[1]
        jac = np.matrix([[-self.L1*np.sin(q1)-self.L2*np.sin(q1+q2), -self.L2*np.sin(q1+q2)],
                         [self.L1*np.cos(q1)+self.L2*np.cos(q1+q2), self.L2*np.cos(q1+q2)]])
        return jac
    
    def DKM(self, q, qdot):
        '''
        Direct Kinematic Model function
        Input : joint positions q=(q1,q2) (in rad)
        Output : end-effector velocity (in m/s)
        '''
        # Use jacobian
        pdot = self.jacobian(q).dot(q)
        return pdot

    def IKM(self, q, pdot):
        '''
        Inverse Kinematic Model function
        Input : end-effector velocity (in m and m/s), joint positions q (in rad)
        Output : joint velocity (rad/s)
        '''
        # Get Jacobian matrix
        jac = self.jacobian(q)
        # Check if Jacobian is singular
        if(np.linalg.det(jac) == 0):
            print("Singular configuration ! ")
        else:
            # Jacobian (inverse)
            qdot = np.linalg.solve(jac, pdot)
            return qdot

class Model1R:
    '''
    Environment class for the 1R robot kinematics + geometry
    '''
    def __init__(self, L=1.):
        ''' 
        Initialize model parameters
        '''
        # Number of joints
        self.nq = 1
        
        # Geometric Model
        self.geometry = GeometricModel1R(L)
        # Kinematic Model
        self.kinematics = KinematicModel1R(L)
        
    def animate(self, q):
        '''
        Animate a given sequence of joint positions
        '''
        self.geometry.animate(q[0,:])

# Calculate desired joint trajectory
class KinematicPlanner:
    
    def __init__(self, model=None):
        '''
        Initialize parameters
        '''
        # Model (geometry + kinematics) 
        self.model = model
        # Number of joints
        self.nq = self.model.nq

    def plan(self, p, pdot):
        '''
        Calculates the desired joint trajectory achieving an input end-effector motion
        Input : end-effector positions, velocities
        Output : joint positions, velocities
        '''
        # Get number of steps
        N = np.shape(p)[1]
        # Initialize joint positions and velocities
        q = np.zeros((self.nq, N))
        qdot = np.zeros((self.nq, N))
        # Get plan using IGM and IKM
        for i in range(N):
            # Use IGM to get joint position
            q[:,i] = self.model.geometry.IGM(p[:,i])
            # Use IKM to get joint velocity
            qdot[:,i] = self.model.kinematics.IKM(q[:,i], pdot[:,i])
        return q, qdot