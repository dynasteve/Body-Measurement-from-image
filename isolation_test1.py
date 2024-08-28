import cv2
import numpy as np

class BodyPartMeasurer:
    def __init__(self, image_path, lower_color1, upper_color1, lower_color2, upper_color2):
        self.image = cv2.imread(image_path)
        self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.lower_color1 = lower_color1
        self.upper_color1 = upper_color1
        self.lower_color2 = lower_color2
        self.upper_color2 = upper_color2
        self.binary_mask = None
        self.contour = None

    def create_binary_mask(self):
        mask1 = cv2.inRange(self.hsv_image, self.lower_color1, self.upper_color1)
        mask2 = cv2.inRange(self.hsv_image, self.lower_color2, self.upper_color2)
        combined_mask = cv2.bitwise_or(mask1, mask2)
        _, self.binary_mask = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)
        return self.binary_mask

    def find_largest_contour(self):
        contours, _ = cv2.findContours(self.binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            self.contour = sorted_contours[0]  # Assuming the largest contour is the target
            return self.contour
        else:
            return None

    def measure_height(self):
        x, y, w, h = cv2.boundingRect(self.contour)
        return h
    
    def measure_height_with_angle(self):
        if self.contour is None:
            return None

        # Get the minimum and maximum points on the y-axis for the contour
        topmost = tuple(self.contour[self.contour[:, :, 1].argmin()][0])
        bottommost = tuple(self.contour[self.contour[:, :, 1].argmax()][0])

        # Calculate the Euclidean distance (to account for the arm's angle)
        height = int(np.sqrt((bottommost[0] - topmost[0]) ** 2 + (bottommost[1] - topmost[1]) ** 2))

        # Draw the topmost and bottommost points and the line representing the height
        cv2.circle(self.image, topmost, 5, (0, 255, 0), -1)
        cv2.circle(self.image, bottommost, 5, (0, 255, 0), -1)
        cv2.line(self.image, topmost, bottommost, (255, 0, 0), 2)

        return height

    def find_width_at_y(self, y, x_center):
        left_length = 0
        right_length = 0
        for x in range(x_center, 0, -1):
            if self.binary_mask[y, x] == 255:
                left_length += 1
            else:
                break
        for x in range(x_center, self.binary_mask.shape[1]):
            if self.binary_mask[y, x] == 255:
                right_length += 1
            else:
                break
        return left_length + right_length

    def measure_widths(self):
        if self.contour is None:
            return None

        x, y, w, h = cv2.boundingRect(self.contour)

        # Calculate y positions for shoulder, chest, belly, and waist
        y_positions = {
            "Shoulder": y + int(0.10 * h),
            "Chest": y + int(0.45 * h),
            "Belly": y + int(0.75 * h),
            "Waist": y + int(0.90 * h)
        }
        
        x_center = x + w // 2
        widths = {}
        
        for key, y_pos in y_positions.items():
            widths[key] = self.find_width_at_y(y_pos, x_center)

        return widths

    def draw_measurements(self, widths):
        x, y, w, h = cv2.boundingRect(self.contour)
        x_center = x + w // 2
        
        for label, y_pos in zip(widths.keys(), [y + int(0.10 * h), y + int(0.45 * h), y + int(0.75 * h), y + int(0.90 * h)]):
            width = widths[label]
            cv2.line(self.image, (x_center - width // 2, y_pos), (x_center + width // 2, y_pos), (255, 0, 0), 2)
            cv2.putText(self.image, f"{label}: {width}px", (x_center + width // 2 + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Body Part Measurements', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage:
if __name__ == "__main__":
    # Define the red color range in HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create an instance of the BodyPartMeasurer
    measurer = BodyPartMeasurer('car.jpg', lower_red1, upper_red1, lower_red2, upper_red2)

    # Create the binary mask
    measurer.create_binary_mask()

    # Find the largest contour (assumed to be the torso)
    measurer.find_largest_contour()

    # Measure the height of the torso
    torso_height = measurer.measure_height()
    print(f"Torso Height: {torso_height} pixels")

    # Measure the widths at specific y-positions
    widths = measurer.measure_widths()
    if widths:
        for key, width in widths.items():
            print(f"{key} Width: {width} pixels")

    # Draw and display the measurements on the image
    measurer.draw_measurements(widths)
