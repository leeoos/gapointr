import numpy as np


class Quaternion:
    """
    Defines a Quaternion class with various operations.

    Attributes:
        w (float): Real part of the quaternion.
        x (float): Imaginary part along the x-axis.
        y (float): Imaginary part along the y-axis.
        z (float): Imaginary part along the z-axis.

    Methods:
        __repr__(): Returns a string representation of the quaternion.
        norm(): Computes the norm (magnitude) of the quaternion.
        normalize(): Normalizes the quaternion.
        conjugate(): Computes the conjugate of the quaternion.
        multiply(other): Multiplies the quaternion by another quaternion.
        quaternion_to_numpy(): Converts the quaternion to a NumPy array.
    """
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Quaternion({self.w}, {self.x}i, {self.y}j, {self.z}k)"

    def norm(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        n = self.norm()
        return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n)

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def multiply(self, other):
        w = self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z
        x = self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y
        y = self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x
        z = self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
        return Quaternion(w, x, y, z)

    def quaternion_to_numpy(self):
        return np.array([self.w, self.x, self.y, self.z])


def quaternion_from_angle_axis(alpha: float, axis: np.ndarray) -> Quaternion:
    """
    Creates a quaternion from an angle-axis representation.

    Parameters:
        alpha (float): Angle of rotation in degrees.
        axis (np.ndarray): Axis of rotation as a NumPy array.

    Returns:
        (Quaternion): Quaternion representing the rotation.
    """
    axis = axis / np.linalg.norm(axis)
    alpha_rad = np.radians(alpha)
    w = np.cos(alpha_rad / 2.0)
    x, y, z = axis * np.sin(alpha_rad / 2.0)
    return Quaternion(w, x, y, z)


def rotate_point(point: np.ndarray, quaternion: Quaternion) -> np.ndarray:
    """
    Rotates a 3D point using a quaternion.

    Parameters:
        point (np.ndarray): 3D point as a NumPy array.
        quaternion (Quaternion): Quaternion representing the rotation.

    Returns:
        (np.ndarray): Rotated 3D point as a NumPy array.
    """

    point_quaternion = Quaternion(0, *point)
    rotated_quaternion = quaternion.multiply(point_quaternion).multiply(quaternion.conjugate())
    return np.array([rotated_quaternion.x, rotated_quaternion.y, rotated_quaternion.z])