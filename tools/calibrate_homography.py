import cv2
import numpy as np
import argparse
import sys

class HomographyCalibrator:
    def __init__(self, source1, source2, id1="0", id2="1"):
        self.cap1 = cv2.VideoCapture(source1)
        self.cap2 = cv2.VideoCapture(source2)
        self.id1 = id1
        self.id2 = id2
        
        if not self.cap1.isOpened() or not self.cap2.isOpened():
            print("Error: Could not open one or both video sources.")
            sys.exit(1)

        # Read the first frame from both
        self.ret1, self.frame1 = self.cap1.read()
        self.ret2, self.frame2 = self.cap2.read()

        if not self.ret1 or not self.ret2:
            print("Error: Could not read frames from sources.")
            sys.exit(1)

        # --- FIX: Standardize Display Size ---
        # Resize both images to a fixed resolution for display.
        # This prevents huge windows for 4K inputs or tiny ones for low-res inputs.
        self.display_size = (800, 600) # (width, height)
        
        # Calculate scale factors for Camera 1
        h1, w1 = self.frame1.shape[:2]
        self.scale_x1 = self.display_size[0] / w1
        self.scale_y1 = self.display_size[1] / h1
        
        # Calculate scale factors for Camera 2
        h2, w2 = self.frame2.shape[:2]
        self.scale_x2 = self.display_size[0] / w2
        self.scale_y2 = self.display_size[1] / h2
        
        self.points1 = []
        self.points2 = []
        
        self.window1 = "Camera 1 (Source)"
        self.window2 = "Camera 2 (Target)"

    def mouse_callback1(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Map display coordinates back to original resolution using precise scale factors
            real_x = int(x / self.scale_x1)
            real_y = int(y / self.scale_y1)
            self.points1.append((real_x, real_y))
            print(f"Cam 1 Point Added: {real_x}, {real_y}")

    def mouse_callback2(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Map display coordinates back to original resolution using precise scale factors
            real_x = int(x / self.scale_x2)
            real_y = int(y / self.scale_y2)
            self.points2.append((real_x, real_y))
            print(f"Cam 2 Point Added: {real_x}, {real_y}")

    def run(self):
        print("\n=== Homography Calibration Tool ===")
        print("Instructions:")
        print("1. Click on a landmark in 'Camera 1'.")
        print("2. Click on the EXACT SAME landmark in 'Camera 2'.")
        print("3. Repeat for at least 4 points (Spread them out across the ground plane).")
        print("4. Press 'c' to Calculate and Print the Matrix.")
        print("5. Press 'r' to Reset points.")
        print("6. Press 'q' to Quit.")
        print("===================================\n")

        cv2.namedWindow(self.window1)
        cv2.namedWindow(self.window2)
        
        cv2.setMouseCallback(self.window1, self.mouse_callback1)
        cv2.setMouseCallback(self.window2, self.mouse_callback2)

        while True:
            # Create display copies resized to FIXED TARGET SIZE
            d1 = cv2.resize(self.frame1, self.display_size)
            d2 = cv2.resize(self.frame2, self.display_size)

            # Draw points on Cam 1
            for i, pt in enumerate(self.points1):
                # Scale real point to display
                px = int(pt[0] * self.scale_x1)
                py = int(pt[1] * self.scale_y1)
                cv2.circle(d1, (px, py), 5, (0, 0, 255), -1)
                cv2.putText(d1, str(i+1), (px+10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw points on Cam 2
            for i, pt in enumerate(self.points2):
                # Scale real point to display
                px = int(pt[0] * self.scale_x2)
                py = int(pt[1] * self.scale_y2)
                cv2.circle(d2, (px, py), 5, (0, 0, 255), -1)
                cv2.putText(d2, str(i+1), (px+10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw lines connecting sequential points for visualization
            if len(self.points1) > 1:
                # Convert list of tuples to list of lists for numpy
                pts_list = [[p[0] * self.scale_x1, p[1] * self.scale_y1] for p in self.points1]
                pts = np.array(pts_list, np.int32)
                cv2.polylines(d1, [pts], False, (255, 0, 0), 2)
            
            if len(self.points2) > 1:
                pts_list = [[p[0] * self.scale_x2, p[1] * self.scale_y2] for p in self.points2]
                pts = np.array(pts_list, np.int32)
                cv2.polylines(d2, [pts], False, (255, 0, 0), 2)

            cv2.imshow(self.window1, d1)
            cv2.imshow(self.window2, d2)

            key = cv2.waitKey(20) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord('r'):
                self.points1 = []
                self.points2 = []
                print("Points Reset.")

            elif key == ord('c'):
                if len(self.points1) < 4 or len(self.points2) < 4:
                    print("Error: Need at least 4 points in each camera.")
                    continue
                
                if len(self.points1) != len(self.points2):
                    print("Error: Number of points must match!")
                    continue

                # Calculate Homography
                src_pts = np.array(self.points1, dtype=np.float32)
                dst_pts = np.array(self.points2, dtype=np.float32)

                H, status = cv2.findHomography(src_pts, dst_pts)

                print("\n" + "="*40)
                print("SUCCESS! Calculated Homography Matrix:")
                print("="*40)
                print("\nCopy the following line under the `homographies:` key of your cameras YAML file (config/cameras*.yaml):\n")
                
                # Format as Python List of Lists
                matrix_str = "["
                for i, row in enumerate(H):
                    matrix_str += f"[{row[0]:.8f}, {row[1]:.8f}, {row[2]:.8f}]"
                    if i < 2: matrix_str += ", "
                matrix_str += "]"
                
                print(f'  "{self.id1}-{self.id2}": {matrix_str}')
                print("\n" + "="*40 + "\n")
                
                # Visualize Verification (Warp Image 1 to Image 2)
                h, w = self.frame2.shape[:2]
                warped_img1 = cv2.warpPerspective(self.frame1, H, (w, h))
                
                # Overlay
                overlay = cv2.addWeighted(self.frame2, 0.5, warped_img1, 0.5, 0)
                # Resize overlay for display as well
                disp_overlay = cv2.resize(overlay, self.display_size)
                cv2.imshow("Verification (Overlap)", disp_overlay)
                print("Showing verification window. Press any key in the window to continue...")

        self.cap1.release()
        self.cap2.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate Homography between two cameras.")
    parser.add_argument("--src1", type=str, required=True, help="Path or ID for Camera 1 (Source)")
    parser.add_argument("--src2", type=str, required=True, help="Path or ID for Camera 2 (Target)")
    parser.add_argument("--id1", type=str, default="0", help="ID string for Camera 1 (e.g., '0')")
    parser.add_argument("--id2", type=str, default="1", help="ID string for Camera 2 (e.g., '1')")
    
    args = parser.parse_args()
    
    # Handle integer camera inputs
    src1 = int(args.src1) if args.src1.isdigit() else args.src1
    src2 = int(args.src2) if args.src2.isdigit() else args.src2

    calibrator = HomographyCalibrator(src1, src2, args.id1, args.id2)
    calibrator.run()