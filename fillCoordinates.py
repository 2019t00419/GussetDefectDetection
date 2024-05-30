import numpy as np
import time

def fill_coordinates(contour_array):

    new_points = []
    last_x, last_y = contour_array[-1][0]

    for i in range(len(contour_array)):
        next_x, next_y = contour_array[i][0]
        
        # Horizontal case
        if (next_x - last_x < -1 or next_x - last_x > 1) and (next_y - last_y == 0):
            if next_x > last_x:
                new_x = np.arange(last_x + 1, next_x)
            else:
                new_x = np.arange(last_x - 1, next_x, -1)
            new_y = np.full_like(new_x, last_y)
            new_points.extend(zip(new_x, new_y))
        
        # Vertical case
        elif (next_y - last_y < -1 or next_y - last_y > 1) and (next_x - last_x == 0):
            if next_y > last_y:
                new_y = np.arange(last_y + 1, next_y)
            else:
                new_y = np.arange(last_y - 1, next_y, -1)
            new_x = np.full_like(new_y, last_x)
            new_points.extend(zip(new_x, new_y))
        
        # Angled case    
        elif (next_x - last_x < -1 or next_x - last_x > 1) and (next_y - last_y < -1 or next_y - last_y > 1):
            if next_x > last_x:
                new_x = np.arange(last_x + 1, next_x)
            else:
                new_x = np.arange(last_x - 1, next_x, -1)
            if next_y > last_y:
                new_y = np.arange(last_y + 1, next_y)
            else:
                new_y = np.arange(last_y - 1, next_y, -1)

            if len(new_x) == len(new_y):  # Ensure the arrays match in length
                new_points.extend(zip(new_x, new_y))
            else:
                if len(new_x) > len(new_y):
                    new_x = new_x[:len(new_y)]
                else:
                    new_y = new_y[:len(new_x)]
                new_points.extend(zip(new_x, new_y))

        new_points.append((next_x, next_y))
        last_x, last_y = next_x, next_y
    
    new_array = np.array(new_points).reshape(-1, 1, 2) 

    return new_array

# Example usage:
# contour_array = np.array([[[0, 0]], [[0, 10]], [[10, 10]], [[10, 0]]])
# filled_contours = fill_coordinates(contour_array)
