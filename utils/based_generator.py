from abc import ABC, abstractmethod
import random

class BasedGenerator(ABC):
    def random_saturated_color(self, backcolor=False):
        # Генерация случайных значений для R, G, B
        r = random.randint(0, 127)
        g = random.randint(0, 127)
        b = random.randint(0, 127)

        if max(r, g, b) - min(r, g, b) < 50:
            if random.choice([True, False]):
                r = random.choice([0, 127])
            else:
                g = random.choice([0, 127])

        return (r, g, b) if not backcolor else (255 - r // 4, 255 - g // 4, 255 - b // 4)

    # def random_position_with_constraints(self):
    #     x_interval, y_interval = self.intervals
    #     x = random.randint(x_interval[0], x_interval[1])
    #     y = random.randint(y_interval[0], y_interval[1])
    #     return (x, y)

    @abstractmethod
    def draw_font(self):
        pass
