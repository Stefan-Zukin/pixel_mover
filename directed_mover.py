import random
import numpy as np
from PIL import Image
import time
import subprocess


class pixel():

    def __init__(self, rgb, row_position, column_position):
        self.rgb_array = rgb
        self.row = row_position
        self.col = column_position
        self.sum = sum(self.rgb_array)

    def rgb(self):
        return self.rgb_array

    def clean(self):
        for c in self.rgb_array:
            if c > 255:
                c = 255
            if c < 0:
                c = 0

    def set_goal(self, row, col):
        self.goal = np.array([row, col])

    def set_position(self, row, col):
        self.row = row
        self.col = col

    def get_step_goal(self, step_size=1):
        step_row = self.row
        step_col = self.col
        if self.goal[0] > self.row + step_size:
            step_row += step_size
        elif self.goal[0] < self.row - step_size:
            step_row -= step_size
        else:
            step_row = self.goal[0]
            
        if self.goal[1] > self.col + step_size:
            step_col += step_size
        elif self.goal[1] < self.col - step_size:
            step_col -= step_size
        else:
            step_col = self.goal[1]
        return [step_row, step_col]

    def at_goal(self):
        if self.row == self.goal[0] and self.col == self.goal[1]:
            return 1
        else:
            return 0

    def distance_to_goal(self):
        return abs(self.row - self.goal[0]) + abs(self.col - self.goal[1])

    def __repr__(self):
        return str([self.row, self.col, list(self.rgb_array)])


class pixel_dictionary():

    def __init__(self):
        self.dict = {}
        self.sorted_keys = []

    def add(self, key, value):
        if(key in self.dict):
            self.dict[key].append(value)
        else:
            self.dict[key] = [value]

    def keys(self):
        return sorted(self.dict.keys())

    def values(self):
        return self.dict.values()

    def get(self, key):
        return self.dict[key]

    def _measure(self, pixel):
        measure = pixel.row + pixel.col - self.measure
        return measure

    def lowest(self, origin=None):
        if self.sorted_keys == []:
            self.sorted_keys = sorted(self.dict.keys())
        key = self.sorted_keys[0]
        values = self.dict[key]
        if origin:
            self.origin = origin
            self.measure = self.origin.row + self.origin.col
            sorted_values = sorted(values, key=self._measure)
            pixel = sorted_values.pop(0)
            values.remove(pixel)
        else:
            pixel = values.pop()
        if(len(values) == 0):
            del self.dict[key]
            self.sorted_keys = self.sorted_keys[1:]
        return pixel

    def __repr__(self):
        return str(self.dict)


class image():
    """Could keep two arrays to move the greyscale values then bring back the colored ones to save"""

    def __init__(self, path, goal_path, smart_search=False):
        img = Image.open(path)
        img.load()
        img_array = np.asarray(img, dtype="int32")
        self.dim = len(img_array)
        self.total_distance = 0
        self.pixels = self._to_pixels(img_array)
        self.dict = pixel_dictionary()
        self.num_at_goal = 0
        self.smart_search = smart_search
        for r in self.pixels:
            for p in r:
                self.dict.add(p.sum, p)
        print("Loaded image")

        goal = Image.open(goal_path)
        goal.load()
        goal_array = np.asarray(goal, dtype="int32")
        self.goal = self._to_pixels(goal_array)
        self.goal_dict = pixel_dictionary()
        for r in self.goal:
            for p in r:
                self.goal_dict.add(p.sum, p)
        print("Loaded goal")

        for k in self.dict.keys():
            random.shuffle(self.dict.get(k))
        print("Shuffled image")

        for k in self.goal_dict.keys():
            random.shuffle(self.goal_dict.get(k))
        print("Shuffled goal")

        it = 0
        for _ in range(self.dim * self.dim):
            pixel = self.dict.lowest()
            if self.smart_search:
                it += 1
                goal = self.goal_dict.lowest(pixel)
                print("Smart matched ", it, "/", self.dim ** 2, "pixels", end="\r")
            else:
                goal = self.goal_dict.lowest()
            pixel.set_goal(goal.row, goal.col)

        self.active_pixels = []
        for r in self.pixels:
            for p in r:
                if p.at_goal():
                    self.num_at_goal += 1
                else:
                    self.active_pixels.append(p)
                    self.total_distance += p.distance_to_goal()
        self.random_list = list.copy(self.active_pixels)
        random.shuffle(self.random_list)


    def _to_pixels(self, array):
        """Converts array of rgb tupes to array of pixel objects"""
        result = np.zeros((self.dim, self.dim), dtype=pixel)
        for row in range(self.dim):
            for col in range(self.dim):
                rgb = array[row][col]
                result[row][col] = pixel(rgb, row, col)
        return result

    def _to_array(self):
        """Converts array of pixel objects to array of rgb tuples"""
        result = np.zeros((self.dim, self.dim, 3))
        for row in range(self.dim):
            for col in range(self.dim):
                pixel = self.pixels[row][col]
                result[row][col] = pixel.rgb()
        return result

    def get_pixel(self, row, col):
        """Returns the pixel object at row, col"""
        return self.pixels[row][col]

    def get_random_pixel(self):
        """Returns a random pixel from the image"""
        return self.get_pixel(random.randint(0, self.dim-1), random.randint(0, self.dim-1))

    def set_pixel(self, row, col, rgb_tuple):
        """Sets a pixel to the given rgb tuple value"""
        pixel = self.pixels[row][col]
        pixel.rgb = rgb_tuple
        pixel.clean()

    def change_pixel(self, row, col, rgb_tuple):
        """Changes a pixel by the given rgb_tuple value"""
        pixel = self.pixels[row][col]
        pixel.rgb_array += rgb_tuple
        pixel.clean()

    def swap(self, pixel1, pixel2):
        """Swaps two pixels"""
        row1 = pixel1.row
        col1 = pixel1.col
        row2 = pixel2.row
        col2 = pixel2.col
        self.pixels[row1][col1] = pixel2
        self.pixels[row2][col2] = pixel1
        pixel1.set_position(row2, col2)
        pixel2.set_position(row1, col1)

    def _fake_goal(self):
        """Returns something close to the goal image
        useful to evaluate whether convergence is true
        """
        for _ in range(10):
            for row in range(self.dim):
                for col in range(self.dim):
                    pixel = self.get_pixel(row, col)
                    goal = self.get_pixel(pixel.goal[0], pixel.goal[1])
                    self.swap(pixel, goal)

    def show(self):
        """Display the image"""
        img = Image.fromarray(self._to_array().astype('uint8'), 'RGB')
        img.show()

    def save(self, name):
        """Save the image to a .png file"""
        img = Image.fromarray(self._to_array().astype('uint8'), 'RGB')
        img.save('output/' + name + '.png')

    def pixel_step(self, pixel, step_size):
        """For the input pixel, make a step towards the goal"""
        if pixel.at_goal():
            return None
        target = pixel.get_step_goal(step_size)
        target_pixel = self.get_pixel(target[0], target[1])
        num_perfect = pixel.at_goal() + target_pixel.at_goal()
        self.swap(pixel, target_pixel)
        post_swap_num_perfect = pixel.at_goal() + target_pixel.at_goal()
        self.num_at_goal += post_swap_num_perfect - num_perfect

    def random_scan(self, step_size=1):
        """Perform one step towards the goal for each pixel in the image in a random order"""
        new_total_distance = self.total_distance
        while new_total_distance >= self.total_distance:
            new_total_distance = 0
            random.shuffle(self.random_list)
            for pixel in self.random_list:
                self.pixel_step(pixel, step_size)
                new_total_distance += pixel.distance_to_goal()
            with open('output.csv', 'a') as f:
                f.write(str(new_total_distance) + ',')
        self.total_distance = new_total_distance
        return (self.dim ** 2) - self.num_at_goal

    def adaptive_step_size(self, it, rate):
        return int(it / (1000 / rate)) + 1

    def step(self, step_size=1):
        """Perform one step towards the goal for a single random pixel"""
        pixel = self.get_random_pixel()

        """This makes it so iterations near convergence won't make no progress
        You can remove it and it will still work, but be slower. It can look
        nicer as it takes longer near convergence to converge."""
        while pixel.at_goal():
            pixel = self.get_random_pixel()

        target = pixel.get_step_goal(step_size)
        target_pixel = self.get_pixel(target[0], target[1])
        num_perfect = pixel.at_goal() + target_pixel.at_goal()
        if num_perfect == 2:
            return
        self.swap(pixel, target_pixel)
        post_swap_num_perfect = pixel.at_goal() + target_pixel.at_goal()
        self.num_at_goal += post_swap_num_perfect - num_perfect
        if self.num_at_goal == self.dim ** 2:
            return 0

    def scan_step(self, step_size=1):
        """Perform one step towards the goal for each pixel in the image"""
        for r in range(self.dim):
            for c in range(self.dim):
                # Change to r,c for vertical lines
                pixel = self.get_pixel(r,c)
                pixel_step(pixel, step_size)
                if self.num_at_goal == self.dim ** 2:
                    return 0

    def active_scan(self, step_size=1):
        """Perform one step towards the goal for each active pixel in the image"""
        for pixel in self.active_pixels:
            target = pixel.get_step_goal(step_size)
            target_pixel = self.get_pixel(target[0], target[1])
            num_perfect = pixel.at_goal() + target_pixel.at_goal()
            self.swap(pixel, target_pixel)
            post_swap_num_perfect = 0
            if pixel.at_goal():
                post_swap_num_perfect += 1
                self.active_pixels.remove(pixel)
            if target_pixel.at_goal():
                post_swap_num_perfect += 1
                self.active_pixels.remove(target_pixel)
            elif target_pixel not in self.active_pixels:
                self.active_pixels.append(target_pixel)
            self.num_at_goal += post_swap_num_perfect - num_perfect
            if self.num_at_goal == self.dim ** 2:
                return 0

    def active_step(self, step_size=1):
        """Perform one step towards the goal using only active pixels"""
        pixel = self.active_pixels.pop(random.randint(0, len(self.active_pixels) - 1))
        target = pixel.get_step_goal(step_size)
        target_pixel = self.get_pixel(target[0], target[1])
        num_perfect = pixel.at_goal() + target_pixel.at_goal()
        self.swap(pixel, target_pixel)
        post_swap_num_perfect = 0
        if pixel.at_goal():
            post_swap_num_perfect += 1
        else:
            self.active_pixels.append(pixel)
        if target_pixel.at_goal():
            post_swap_num_perfect += 1
        elif target_pixel not in self.active_pixels:
            self.active_pixels.append(target_pixel)
        self.num_at_goal += post_swap_num_perfect - num_perfect
        if self.num_at_goal == self.dim ** 2:
            return 0


def main(initial_path, goal_path, step_size, save_step, smart_search=True, scan=False, active=False, point=False):
    i = image(path=initial_path, goal_path=goal_path, smart_search=smart_search)
    i.save('00000')

    it = 0
    t1 = time.time()
    time_remaining = '???'
    remaining_points = 1
    stalled = 0

    #Adaptive Step
    if step_size == 0:
        step_size = i.adaptive_step_size(it, adaptation_rate)
    
    while remaining_points != 0:
        if active and scan:
            remaining_points = i.active_scan(step_size)
        elif random_scan:
            remaining_points = i.random_scan(step_size)
        elif scan:
            remaining_points = i.scan_step(step_size)
        elif active:
            remaining_points = i.active_step(step_size)
        else:
            remaining_points = i.step(step_size)
        it += 1
        print("Iteration:", it, "Pixels Remaining:", remaining_points, "Progress %:", round(100 * i.num_at_goal/(i.dim ** 2), 1),
            "Estimated Time Remaining:", time_remaining, end='\r')

        # Actions to execute on save step
        if it % save_step == 0:
            i.save(str(int(it/save_step)).zfill(5))
            time_elapsed = time.time() - t1
            time_remaining = time.strftime('%H:%M:%S', time.gmtime(
                time_elapsed / (i.num_at_goal/(i.dim ** 2)) - time_elapsed))

        #Check if stalled
        # if remaining_points < min_remaining_points:
        #     min_remaining_points = remaining_points
        #     stalled = 0
        # else:
        #     stalled += 1
        # if stalled > 20:
        #     print("Stopping early") 
        #     break

        # past_remaining_points = remaining_points

    #Actions to execute on convergence    
    t2 = time.time()
    print("\nFinished in", time.strftime(
        '%H:%M:%S', time.gmtime(t2-t1)))
    i.save(str(int(it/save_step)).zfill(5))


if __name__ == "__main__":
    initial_path = 'images/winter_128.png'
    goal_path = 'images/wave_128.png'
    adaptation_rate = 2  #Updates step size every 1000/adaptation_rate frames. 
    step_size = 1  #0 for adaptive step size. Boosts speed for larger images. Step size will start small and increase as iterations progress.
    save_step = 1 #Save an image every _ frames
    smart_search = False
    random_scan = True #Move pixels in a random order
    scan = False #Move pixels by scanning across the image. Faster, but gives a different effect.
    active = False
    main(initial_path, goal_path, step_size, save_step, smart_search, scan, active, random_scan)
    try:
        subprocess.run("ffmpeg -r 30 -f image2 -s 1920x1080 -i output/%05d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p output/movie.mp4", shell=True)
    except:
        print("Rendering frames into movie failed. Check that you have ffmpeg installed")
        print("Try manually running the command from the output folder: ffmpeg -r 30 -f image2 -s 1920x1080 -i %05d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p movie.mp4")

# IDeas:
# It can be like diffusion. Reaching energy minima. Compute a potential energy surface based on the pixel darknesses in the goal
# Then you can do something like gradient descent for each pixel to find its minima
# Check if a pixel is where it started last time