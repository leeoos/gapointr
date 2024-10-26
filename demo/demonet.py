import torch
import numpy as np
import open3d as o3d
import torch.nn as nn
import matplotlib.pyplot as plt

from clifford_modules.mvlinear import MVLinear
from clifford_modules.mvrelu import MVReLU

# Generate clifford algebra
from clifford_lib.algebra.cliffordalgebra import CliffordAlgebra


class PointCloudDeformationNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3):
        super(PointCloudDeformationNet, self).__init__()
        self.name = 'deformer'
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # Output the deformed point coordinates
        )

    def forward(self, x):
        return self.network(x)
    

class PointCloudGADeformationNet(nn.Module):
    def __init__(self, algebra, input_dim=3, hidden_dim=64, output_dim=3):
        super(PointCloudGADeformationNet, self).__init__()
        self.name = 'ga_deformer'
        
        # self.cgemlp = CGEBlock(algebra, input_dim, hidden_dim)
        self.mlp = nn.Sequential(
            MVLinear(algebra, input_dim, hidden_dim),
            MVReLU(algebra, hidden_dim),
            MVLinear(algebra, hidden_dim, hidden_dim),
            MVReLU(algebra, hidden_dim),
            MVLinear(algebra, hidden_dim, output_dim)
        )
        # Projecting multivectors to points
        self.prj = nn.Linear(in_features=2**algebra.dim, out_features=1)  

    def forward(self, input):
        h = self.mlp(input)
        # Index the hidden states at 0 to get the invariants, and let a regular MLP do the final processing.
        print(h.shape)
        return self.prj(h).squeeze()
    
def load_point_cloud(pcd_file):
    # Load the point cloud data from a PCD file
    pcd = o3d.io.read_point_cloud(pcd_file)
    # Convert to NumPy array for easier manipulation
    points = np.asarray(pcd.points)
    return points


def deform_point_cloud(model, point_cloud, device='cpu'):
    point_cloud = torch.tensor(point_cloud, dtype=torch.float32).to(device)
    if model.name == 'ga_deformer': point_cloud = point_cloud.unsqueeze(-1)
    model = model.to(device)
    param_device = next(model.parameters()).device
    print(f"Model device: {param_device}")
    print(f"PNC device: {point_cloud.device}")
    assert param_device == point_cloud.device
    with torch.no_grad():
        deformed_points = model(point_cloud).cpu().numpy()
    return deformed_points


def save_deformed_point_cloud(deformed_points, output_file):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(deformed_points)
    o3d.io.write_point_cloud(output_file, pcd)


def point_cloud_to_image_with_color(point_cloud, img_size=(256, 256), point_size=1, output_file="output_image.png"):
    """
    Converts a 3D point cloud to a 2D image using orthographic projection and Z-value for intensity, then saves it.
    
    Parameters:
    - point_cloud: NumPy array of shape (N, 3), where N is the number of points.
    - img_size: Tuple representing the size of the output image (width, height).
    - point_size: Size of each point in the image.
    - output_file: The file path to save the image.
    
    Returns:
    - None
    """
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    # Normalize the x, y, and z coordinates
    x_normalized = (x - np.min(x)) / (np.max(x) - np.min(x)) * (img_size[0] - 1)
    y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y)) * (img_size[1] - 1)
    z_normalized = (z - np.min(z)) / (np.max(z) - np.min(z))

    # Create a blank image
    img = np.zeros((img_size[1], img_size[0]))

    # Project each point into 2D image and use z for intensity
    for i in range(len(x)):
        xi = int(x_normalized[i])
        yi = int(y_normalized[i])
        img[yi, xi] = z_normalized[i]  # Use the z value for grayscale intensity

    # Save the resulting image
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='plasma')  # Color map 'plasma' for better visualization
    plt.axis('off')  # No axis for a clean look

    # Save the image to the specified output file
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to avoid display
    print(f"Image saved as {output_file}")



if __name__ == '__main__':

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # Load the original point cloud
    input_pcd = "./demo/airplane.pcd"
    points = load_point_cloud(input_pcd)
    print(f"Point types: {type(points)}")
    print(f"Points shape: {points.shape}")

    # Build algebra
    algebra_dim = int(points.shape[1])
    metric = [1 for i in range(algebra_dim)]
    print("\nGenerating the algebra...")
    algebra = CliffordAlgebra(metric)
    algebra
    print(f"algebra dimention: \t {algebra.dim}")
    print(f"multivectors elements: \t {sum(algebra.subspaces)}")
    print(f"number of subspaces: \t {algebra.n_subspaces}")
    print(f"subspaces grades: \t {algebra.grades.tolist()}")
    print(f"subspaces dimentions: \t {algebra.subspaces.tolist()}")
    print("done")

    # Define your model
    # model = PointCloudDeformationNet()
    model = PointCloudGADeformationNet(algebra)

    print("processing!")

    # Deform the point cloud (inference after training)
    deformed_points = deform_point_cloud(model, points, device=device)
    print(type(deformed_points))
    print(deformed_points.shape)
    

    # Save the deformed point cloud
    output_pcd = "deformed_output.pcd"
    save_deformed_point_cloud(deformed_points, output_pcd)
    

    points = load_point_cloud(output_pcd)

    # Convert point cloud to image
    point_cloud_to_image_with_color(points, img_size=(256, 256), output_file="colored_cloud.png")

    print("done!")
    





# class CliffordTransformationNet(nn.Module):
#     def __init__(self, input_dim=3, hidden_dim=64):
#         super(CliffordTransformationNet, self).__init__()

#         # Simple MLP to predict rotation angles and translation vector
#         self.network = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 4)  # Outputs 1 rotation angle and 3 translation values
#         )

#     def forward(self, x):
#         """
#         Forward pass where Clifford algebra transformations (rotations, translations) 
#         are applied directly inside the neural network.
        
#         Parameters:
#         - x: Input point cloud of shape (N, 3).
        
#         Returns:
#         - Transformed point cloud of shape (N, 3).
#         """
#         # Predict transformation parameters from the point cloud
#         transformation_params = self.network(x.mean(dim=0))  # Aggregate input to produce a single set of transformations

#         # Extract predicted parameters
#         rotation_angle = transformation_params[0]  # Rotation angle in radians
#         translation_vector = transformation_params[1:]  # Translation vector [tx, ty, tz]

#         # Convert to NumPy for Clifford algebra operations
#         points = x.detach().numpy()

#         # Apply Clifford algebra-based rotation and translation
#         transformed_points = self.apply_clifford_transform(points, rotation_angle.item(), translation_vector.detach().numpy())

#         # Return transformed points as a torch tensor
#         return torch.tensor(transformed_points, dtype=torch.float32)


#     def apply_clifford_transform(self, points, rotation_angle, translation_vector, axis='z'):
#         """
#         Applies a Clifford algebra-based rotation and translation to a point cloud.
        
#         Parameters:
#         - points: NumPy array of shape (N, 3) representing the point cloud.
#         - rotation_angle: Rotation angle in radians.
#         - translation_vector: 3D translation vector.
#         - axis: Axis for rotation ('x', 'y', or 'z').
        
#         Returns:
#         - Transformed points: NumPy array of shape (N, 3).
#         """
#         # Perform rotation using Clifford algebra
#         rotated_points = self.rotate_point_cloud(points, rotation_angle, axis)

#         # Perform translation
#         translated_points = self.translate_point_cloud(rotated_points, translation_vector)

#         return translated_points

#     def rotate_point_cloud(self, points, rotation_angle, axis='z'):
#         """
#         Rotates a point cloud using Clifford algebra.
        
#         Parameters:
#         - points: NumPy array of shape (N, 3), where N is the number of points.
#         - rotation_angle: The angle in radians to rotate.
#         - axis: The axis to rotate around (e.g., 'x', 'y', 'z').
        
#         Returns:
#         - Rotated points: NumPy array of shape (N, 3).
#         """
#         # Define rotation axis as a bivector in Clifford algebra
#         if axis == 'x':
#             rotation_bivector = e2 * e3  # rotation around x-axis
#         elif axis == 'y':
#             rotation_bivector = e3 * e1  # rotation around y-axis
#         elif axis == 'z':
#             rotation_bivector = e1 * e2  # rotation around z-axis
#         else:
#             raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")

#         # Create the rotor for rotation
#         rotor = np.cos(rotation_angle / 2) + np.sin(rotation_angle / 2) * rotation_bivector

#         # Apply the rotor to each point in the point cloud
#         rotated_points = []
#         for point in points:
#             vec = point[0] * e1 + point[1] * e2 + point[2] * e3
#             rotated_vec = rotor * vec * ~rotor
#             rotated_points.append([rotated_vec[e1], rotated_vec[e2], rotated_vec[e3]])

#         return np.array(rotated_points)

#     def translate_point_cloud(self, points, translation_vector):
#         """
#         Translates a point cloud.
        
#         Parameters:
#         - points: NumPy array of shape (N, 3), where N is the number of points.
#         - translation_vector: A 3D translation vector [tx, ty, tz].
        
#         Returns:
#         - Translated points: NumPy array of shape (N, 3).
#         """
#         return points + np.array(translation_vector)