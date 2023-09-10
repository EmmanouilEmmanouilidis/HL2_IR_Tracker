import numpy as np
import cv2
import copy
import open3d as o3d
import itertools
import heapq


class HL2Calibration:
    def __init__(self):
        ir_img1 = cv2.imread("1667909862_abImage.tiff", cv2.IMREAD_GRAYSCALE)
        d_img1 = cv2.imread("1667909862_depth.tiff", cv2.IMREAD_GRAYSCALE)

        self.ir_images = [ir_img1] 
        self.d_images = [d_img1] 

    def controller(self, img, brightness=255, contrast=127):
        brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
        contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
    
        if brightness != 0:
    
            if brightness > 0:
                shadow = brightness
                max = 255
            else:
                shadow = 0
                max = 255 + brightness
            al_pha = (max - shadow) / 255
            ga_mma = shadow
            cal = cv2.addWeighted(img, al_pha, img, 0, ga_mma)
        else:
            cal = img
    
        if contrast != 0:
            Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
            Gamma = 127 * (1 - Alpha)
            cal = cv2.addWeighted(cal, Alpha, cal, 0, Gamma)
        
        return cal

    def get_uv_map(self):
        with open("uvMap.txt") as f:
            lines = f.readlines()
        np_map = np.array(lines).reshape((512,512,2)).astype(float)

        return np_map

    def get_depth_image(self, depth_img):
        new_img = depth_img  # self.controller(depth_img, 333, 175)

        return np.asarray(new_img)

    def detect_keypoints(self, img):
        tmp_ir_img = img 
        ir_img = self.controller(tmp_ir_img, 378, 248)
        data = np.asarray(ir_img)
        _, th = cv2.threshold(data, 160, 255, cv2.THRESH_BINARY)
        resized = th #cv2.resize(th, dims, interpolation=cv2.INTER_LINEAR)

        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200
        #params.minDistBetweenBlobs = 0

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 2

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.87

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01

        # |Filter by Color
        params.filterByColor = False
        params.blobColor = 255

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(resized)

        return keypoints

    def draw_registration_result(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])
    
    def find_basis(self, points):
        candidate_dict = {}
        for p_id, point in enumerate(points):
            neighbors = [points[i] for i in range(len(points)) if i != p_id]
            distances = []
            ref_dict = {}
            for neighbor in neighbors:    
                dist = np.linalg.norm(neighbor-point)
                ref_dict[dist] = neighbor
                distances.append(dist)
            candidates = heapq.nlargest(2, distances)
            first_vec, second_vec = ref_dict.get(candidates[0]) - point, ref_dict.get(candidates[1]) - point
            candidate_dict[p_id] = [first_vec, second_vec]
        
        point_bases = []
        while len(candidate_dict.keys()) > 0:
            candidate_id = -1
            first_vec_c, second_vec_c = None, None
            first_v_length_c, second_v_length_c = 0.0, 0.0

            for id in candidate_dict.keys():            
                v1, v2 = candidate_dict[id]
                l_v1 = np.linalg.norm(v1)
                l_v2 = np.linalg.norm(v2)

                if (first_v_length_c <= l_v1) and (second_v_length_c <= l_v2):
                    first_vec_c = v1
                    second_vec_c = v2
                    candidate_id = id
                    first_v_length_c = l_v1
                    second_v_length_c = l_v2

            z_vec = np.cross(first_vec_c, second_vec_c) #[0.0, 0.0, 1.0]
            candidate = [first_vec_c, second_vec_c, z_vec]
            point_bases.append({candidate_id: candidate})
            candidate_dict.pop(candidate_id)

        return candidate, candidate_id, point_bases

    def gram_schmidt(self, A):
        M = A.shape[0]
        N = A.shape[1]
        assert(M >= N)
        for k in range(0,N):
            A[:,k] = A[:,k] - np.dot(A[:,:k], np.dot(A[:,k], A[:,:k]))
            A[:,k] = (1.0 / np.linalg.norm(A[:,k])) * A[:,k]
        return A

    def compute_distances(self, ir_image, depth_image):
        kp_vals = []
        uv_vals = []
        range_vals = []
        depth_vals = []
        coords_3d = []
        distances_detected = []
        distances_ref = [np.array([0.067, 0.0, 0.0]), np.array([0.069, 0.042, 0.0]), np.array([0.036, 0.02, 0]), np.array([0.0, 0.045, 0.0]), np.array([0.035, 0.025, 0.0]), np.array([0.03, 0.03, 0.0])]
        
        distances_ref_large = [np.array([0.165, 0.0, 0.0]), np.array([0.085, 0.04, 0.0]), np.array([0.125, 0.075, 0.0]), np.array([0.08, 0.04, 0.0]), np.array([0.045, 0.075, 0.0]), np.array([0.04, -0.035, 0.0])]

        estimate = []
        truth = [np.array([0.069, 0.0, 0.0]), np.array([-0.069, 0.0, 0.0]), np.array([0.069, 0.042, 0.0]), np.array([-0.069, -0.042, 0.0]), np.array([0.036, 0.02, 0]) ,np.array([-0.036, -0.02, 0]), 
                np.array([0.0, 0.045, 0.0]), np.array([0.0, -0.045, 0.0]), np.array([0.03, 0.025, 0.0]), np.array([-0.03, -0.025, 0.0]), np.array([0.03, 0.03, 0.0]), np.array([-0.03, -0.03, 0.0])]
        all_truths = list(itertools.permutations(distances_ref, len(distances_ref)))

        ref_model = np.array([[0.0, 0.0, 0.0],[0.068, 0.0, 0.0], [0.038, 0.025, 0.0], [0.06, 0.048, 0.0]])
        #pcd = o3d.geometry.PointCloud()
        #pcd.points = o3d.utility.Vector3dVector(ref_model)
        #pcd.paint_uniform_color([1, 0, 0])
        #o3d.visualization.draw_geometries([pcd])

        uv_map = self.get_uv_map()
        keypoints = self.detect_keypoints(ir_image)
        depth_img = self.get_depth_image(depth_image)
        
        for kp in keypoints:
            kp_vals.append([int(kp.pt[0]), int(kp.pt[1])])
        
        for kp in kp_vals:
            uv_vals.append(uv_map[kp[0], kp[1]])
            range_vals.append(float(depth_img[kp[0], kp[1]]))
        
        for idx, r in enumerate(range_vals): 
            d = r / 255 
            d = d * (1000/255)
            depth_vals.append(d)
        

        for idx, d in enumerate(depth_vals):
            tmp_vec = np.array([uv_vals[idx][0], uv_vals[idx][1], -1.0])
            point3d = np.multiply(tmp_vec, d)
            coords_3d.append(point3d)
        coords_3d = np.array(coords_3d)

        #ref_model = np.array([[0.0, 0.0, 0.0],[-0.068, 0.0, 0.0], [-0.038, 0.025, 0.0], [-0.06, 0.048, 0.0]])

        ref_base, ref_origin_id, all_bases_ref = self.find_basis(ref_model)
        det_base, det_origin_id, all_bases_det = self.find_basis(coords_3d)
        
        ref_ids = np.array([[int(b) for b in base] for base in all_bases_ref]).flatten()
        ord_ref = ref_model[ref_ids]

        est_ids = np.array([[int(b) for b in base] for base in all_bases_det]).flatten()
        ord_est = coords_3d[est_ids]

        c, r, t = self.rigid_transform_3D(np.array(ord_est), np.array(ord_ref), True)
        print("")
        print("c: {}".format(c))
        print("r: {}".format(r))
        print("t: {}".format(t))
        print("")
        coords_trans = []
        for e in ord_ref:
            coords_trans.append(c * (r @ e + t))
        coords_trans = np.array(coords_trans)

        print("")
        print("Reference: {}".format(ord_ref))
        print("Estimate: {}".format(ord_est))
        print("Transformed: {}".format(coords_trans))
        print("")

        pcd_ref = o3d.geometry.PointCloud()
        pcd_ref.points = o3d.utility.Vector3dVector(ref_model)
        pcd_ref.paint_uniform_color([1, 0, 0])
        pcd_detect = o3d.geometry.PointCloud()
        pcd_detect.points = o3d.utility.Vector3dVector(ord_est)
        pcd_detect.paint_uniform_color([0, 0, 1])
        pcd_trans = o3d.geometry.PointCloud()
        pcd_trans.points = o3d.utility.Vector3dVector(coords_trans)
        pcd_trans.paint_uniform_color([0, 1, 0])

        o3d.visualization.draw_geometries([pcd_ref, pcd_detect, pcd_trans])


        exit()


        print(det_origin_id)
        print(det_base)
        print("")
        print(all_bases_det)
        
        coord_ref = self.gram_schmidt(np.array(ref_base))
        coord_det = self.gram_schmidt(np.array(det_base))

        print("*"*40)
        print("")
        print(ref_origin_id)
        print(ref_model[ref_origin_id])
        print(ref_base)
        print(det_origin_id)
        print(coords_3d[det_origin_id])
        print(det_base)
        print("")
        print(coord_ref)
        print(coord_det)
        print("")
        print("*"*40)


        mesh_ref = o3d.geometry.TriangleMesh.create_coordinate_frame()
        mesh_ref.scale(0.05,center=ref_model[ref_origin_id])
        mesh_det = o3d.geometry.TriangleMesh.create_coordinate_frame()
        mesh_det.translate(coords_3d[det_origin_id])
        mesh_det.scale(0.05,center=coords_3d[det_origin_id])
        pcd_ref = o3d.geometry.PointCloud()
        pcd_ref.points = o3d.utility.Vector3dVector(ref_model)
        pcd_ref.paint_uniform_color([1, 0, 0])
        pcd_detect = o3d.geometry.PointCloud()
        pcd_detect.points = o3d.utility.Vector3dVector(np.array(coords_3d))
        pcd_detect.paint_uniform_color([0, 0, 1])
        o3d.visualization.draw_geometries([mesh_ref, mesh_det, pcd_ref, pcd_detect])

        for idx, vec in enumerate(coords_3d):
            dists = []
            for n in range(idx+1, len(coords_3d)):
                tmp = np.array([vec, coords_3d[n]])
                if len(tmp) > 0:
                    d = np.abs(vec- coords_3d[n])
                    dists.append(d)
                    estimate.append(d)

        if dists != []:
            distances_detected.append(dists)

        return truth, estimate, all_truths

    # Kabsch algorithm
    def rigid_transform_3D(self, A, B, scale=False):
        N = A.shape[0]

        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)

        AA = A - np.tile(centroid_A, (N, 1))
        BB = B - np.tile(centroid_B, (N, 1))

        if scale:
            H = BB.T @ AA / N
        else:
            H = BB.T @ AA

        U, S, Vt = np.linalg.svd(H)
        R = Vt.T * U.T

        if np.linalg.det(R) < 0:
            print ("Reflection detected")
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        if scale:
            varA = np.var(A, axis=0).sum()
            c = 1 / (1 / varA * np.sum(S)) 
            t = -R @ (centroid_B.T * c) + centroid_A.T
        else:
            c = 1
            t = -R @ centroid_B.T + centroid_A.T

        return c, R, t
    
    def mse(self, x, y):
        #print(x)
        #print(np.array(y))
        return np.square(np.subtract(x, y)).mean()

    def calibrate(self, truth, estimate, all_truths):
        best_fit = None
        lowest_mse = float('inf')
        id = 0
        for idx, candidate in enumerate(all_truths):
            mse_candidate = self.mse(estimate, list(candidate))
            if mse_candidate < lowest_mse:
                lowest_mse = mse_candidate
                best_fit = candidate
                id = idx
        print(id)
        print(best_fit)

        c, r, t = self.rigid_transform_3D(np.array(best_fit), np.array(estimate), False)

        transformed_pts = []
        for e in estimate:
            transformed_pts.append(c * (r @ e + t))
        
        print("")
        print("*" * 20)
        print(best_fit)
        print("")
        print(estimate)
        print("")
        print(transformed_pts)

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(np.asarray(estimate)) 
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(np.asarray(best_fit))

        threshold = 0.002
        trans_init = np.asarray([[0.862, 0.011, -0.507, 0.0],
                                [-0.139, 0.967, -0.215, 0.0],
                                [0.487, 0.255, 0.835, 0.0], 
                                [0.0, 0.0, 0.0, 1.0]])
        #self.draw_registration_result(source, target, trans_init)

        #source_alt = o3d.geometry.PointCloud()
        #source_alt.points = o3d.utility.Vector3dVector(np.asarray(new_estimate)) 
        #self.draw_registration_result(source_alt, target, trans_init)
        #print("Initial alignment")
        #evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
        #print(evaluation)

        #print("Apply point-to-point ICP")
        #reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
        #print(reg_p2p)
        #print("Transformation is:")
        #print(reg_p2p.transformation)
        #print("")
        #self.draw_registration_result(source, target, reg_p2p.transformation)

if __name__ == "__main__":
    estimates = []
    ground_truth = None
    calib = HL2Calibration()
    for idx in range(len(calib.ir_images)):
        ground_truth, estimate, all_truths = calib.compute_distances(calib.ir_images[idx], calib.d_images[idx])
        estimates.append(estimate)
    estimate = np.mean(estimates, axis=0)
    print(estimate)
    calib.calibrate(ground_truth, estimate, all_truths)