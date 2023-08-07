
import numpy as np
import matplotlib.pyplot as plt

class Arc:
    """
    Class describing a linearly interpolated arc 
    """

    def __init__(self, pts):
        """
        Initialize the tangent and normal lines for the arc at the given points
        pts:    (N, 2) numpy array of arc points
        """
        assert len(pts) > 1
        assert pts.shape[-1] == 2
        self.pts = pts      # (N, 2)
        self.n = len(pts)
        self.tangents = []  # list of coeffs (a,b,c) of the tangents ax+by+c=0 at each point
        self.normals = []   # list of coeffs (a,b,c) of the normals ax+by+c=0 at each point

        for i in range(self.n):
            point = self.pts[i]
            secant_pt0 = self.pts[i-1] if i>0 else None
            secant_pt1 = self.pts[i+1] if i<self.n-1 else None
            tangent_coeff = self.fit_tangent(point, secant_pt0, secant_pt1)
            normal_coeff = self.fit_normal(tangent_coeff, point)
            self.tangents.append(tangent_coeff)
            self.normals.append(normal_coeff)


    @staticmethod
    def fit_tangent(point, secant_pt0=None, secant_pt1=None):
        """
        finds the tangent line at 'point' parallel to the secant line given by secant_pt.
        If one of secant_pt is not given, then this tangent line is equivalent to 
        the line that passes through the given two points.
        """
        assert not ((secant_pt0 is None) and (secant_pt1 is None))
        secant_pt0 = point if secant_pt0 is None else secant_pt0
        secant_pt1 = point if secant_pt1 is None else secant_pt1
        (x,y) = point
        (x0,y0) = secant_pt0
        (x1,y1) = secant_pt1
        a = y1-y0
        b = x0-x1
        c = x*(y0-y1) + y*(x1-x0)
        return (a,b,c)


    @staticmethod
    def fit_normal(coeffs, point):
        # (b)x + (-a)y + (ay0-bx0) = 0
        (a,b,c) = coeffs
        (x0,y0) = point
        return (b, -a, (a*y0-b*x0))


    @staticmethod
    def intersection(line1, line2):
        (a1,b1,c1) = line1
        (a2,b2,c2) = line2
        if (a1*b2-a2*b1==0) or (b1*a2-b2*a1==0):
            return None
        x = (b1*c2 - b2*c1)/(a1*b2 - a2*b1)
        y = (a1*c2 - a2*c1)/(b1*a2 - b2*a1)
        return (x,y)


    @staticmethod
    def projection(u, v):
        # scalar projection of (numpy) vectors u onto v
        # cosθ = <u,v>/|u||v|
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)
        if u_norm==0 or v_norm==0:
            return 0, 0
        # clip to account for precision error
        cos_theta = np.clip(abs(u @ v) / (u_norm*v_norm), a_min=0, a_max=1)
        sin_theta = np.sqrt(1-cos_theta**2)
        u_proj_v = u_norm * cos_theta
        u_reject_v = u_norm * sin_theta
        return u_proj_v, u_reject_v



    def deviation(self, points):
        """
        calculates all deviations (lateral and longitudinal) between this arc
        and an array of points (N, 2)
        """
        num_pts = min(self.n, len(points))
        lat_deviation, long_deviation = [], []
        for i in range(num_pts):
            p0 = self.pts[i]    # (2,)
            p1 = points[i]      # (2,)
            (a,b,c) = self.normals[i]
            u = p1 - p0
            v = np.array([b, -a])
            u_proj_v, u_reject_v = self.projection(u, v)
            lat_deviation.append(u_proj_v)
            long_deviation.append(u_reject_v)
        return np.array(lat_deviation), np.array(long_deviation)


    def maxDeviation(self, points):
        """
        calculates the maximum deviation (lateral and longitudinal) between this arc
        and an array of points (N, 2)
        """
        num_pts = min(self.n, len(points))
        max_lat_deviation = max_long_deviation = 0
        for i in range(num_pts):
            p0 = self.pts[i]    # (2,)
            p1 = points[i]      # (2,)
            (a,b,c) = self.normals[i]
            u = p1 - p0
            v = np.array([b, -a])
            u_proj_v, u_reject_v = self.projection(u, v)
            max_lat_deviation = max(u_proj_v, max_lat_deviation)
            max_long_deviation = max(u_reject_v, max_long_deviation)
        return max_lat_deviation, max_long_deviation


    def avgDeviation(self, points):
        """
        calculates the average deviation (lateral and longitudinal) between this arc
        and an array of points (N, 2)
        """
        num_pts = min(self.n, len(points))
        lat_deviation = []
        long_deviation = []
        for i in range(num_pts):
            p0 = self.pts[i]    # (2,)
            p1 = points[i]      # (2,)
            (a,b,c) = self.normals[i]
            u = p1 - p0
            v = np.array([b, -a])
            u_proj_v, u_reject_v = self.projection(u, v)
            lat_deviation.append(u_proj_v)
            long_deviation.append(u_reject_v)
        return sum(lat_deviation)/len(lat_deviation), sum(long_deviation)/len(long_deviation)




# ========================================================================
# =============================== TESTING ================================
# ========================================================================

def test():
    x = np.linspace(-1,1,num=50)
    y0 = np.sqrt(1-x**2)
    y1 = np.zeros_like(x)
    arc0 = np.stack((x,y0), axis=-1)
    arc1 = np.stack((x,y1), axis=-1)

    plt.plot(x, y0, color='green', linestyle='-', linewidth=2)
    plt.plot(x, y1, color='red', linestyle='-', linewidth=2)
    plt.scatter(x, y0, color='blue')
    plt.scatter(x, y1, color='blue')


    arc = Arc(arc0)
    x = np.linspace(-1.25,1.25,num=50)
    for (normal, tangent) in zip(arc.normals, arc.tangents):
        (a,b,c) = normal
        y = (-a*x-c)/b
        plt.plot(x, y, color='orange')

    plt.xlim([-1.25,1.25])
    plt.ylim([-0.25,1.25])
    plt.gca().set_aspect('equal')
    plt.show()

    r,s = arc.maxDeviation(arc1)
    print(r,s)
    # NOTE: correct answer should be (1.0, 0.5)

    r,s = arc.deviation(arc1)
    print(np.max(r),np.max(s))

    r,s = arc.avgDeviation(arc1)
    print(r,s)


if __name__=='__main__':
    test()
    

