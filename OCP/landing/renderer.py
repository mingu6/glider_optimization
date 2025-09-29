import pygame
import numpy as np

class PygameRenderer:
    def __init__(self, states, state_parser, scale=80.0, width=1000, height=600):
        self.state_parser = state_parser
        if callable(states):
            self.states_callable = states
            self.is_generator = True
            self.states_gen = iter(self.states_callable())
        else:
            self.states_list = list(states)
            self.is_generator = False
            self.states_gen = iter(self.states_list)
        self.scale = scale
        self.W, self.H = width, height
        self.index = 0
        self.time_scale = 1.0
        pygame.init()
        self.screen = pygame.display.set_mode((self.W, self.H))
        self.clock = pygame.time.Clock()
        font = pygame.font.get_default_font()
        self.fnt = pygame.font.Font(font, 18)
        self.running = True

    def restart(self):
        self.index = 0
        if self.is_generator:
            self.states_gen = iter(self.states_callable())
        else:
            self.states_gen = iter(self.states_list)

    def draw_frame(self, state):
        self.screen.fill((240,240,240))
        ground_y = self.H - 150
        pygame.draw.line(self.screen, (0,0,0), (0, ground_y), (self.W, ground_y), 3)

        points_list = self.state_parser(state)
        for seg in points_list:
            pts_screen = [(int(x*self.scale + self.W*0.5), int(ground_y - y*self.scale)) for x,y in seg]
            if len(pts_screen) == 1:
                pygame.draw.circle(self.screen, (200,30,30), pts_screen[0], 6)
            else:
                pygame.draw.lines(self.screen, (30,144,255), False, pts_screen, 6)
                for pt in pts_screen:
                    pygame.draw.circle(self.screen, (200,30,30), pt, 6)


        fps = int(self.clock.get_fps())
        txt = self.fnt.render(
            f"time_scale {self.time_scale:.2f}    fps {fps:.2f}",
            True, (10,10,10)
        )
        self.screen.blit(txt, (10,10))
        pygame.display.flip()
    def run(self, fps=60):
        while self.running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.running = False
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_r:
                        self.restart()
                    if ev.key == pygame.K_UP:
                        self.time_scale *= 1.5
                    if ev.key == pygame.K_DOWN:
                        self.time_scale /= 1.5
            try:
                state = next(self.states_gen)
            except StopIteration:
                self.restart()
            self.draw_frame(state)
            self.index += 1
            self.clock.tick(fps * self.time_scale)


if __name__ == "__main__":
    def state_gen():
        x, z, theta = 0.0, 2.0, 0.2
        for i in range(1000):
            x += 0.01
            yield (x, z, theta)

    l = [[0.0, 2.0, 0.2]]  # list of states
    for i in range(1000):
        x, z, theta = l[-1]
        l.append([x + 0.01, z, theta])

    def glider_state_parser(state):
        """
        Convert a generic state into drawable segments.
        Returns a list of segments (each segment is a list of points).
        Currently supports just the fuselage.
        """
        x, z, theta = state[:3]  # first three entries are always x, z, theta
        half_length = 1.0
        points_body = np.array([[-half_length, 0.0], [half_length, 0.0]])
        ct, st = np.cos(theta), np.sin(theta)
        R = np.array([[ct, -st], [st, ct]])
        world_pts = (R @ points_body.T).T + np.array([x, z])
        return [world_pts.tolist()]  # list of one segment

    renderer = PygameRenderer(state_gen, glider_state_parser)
    renderer.run()
