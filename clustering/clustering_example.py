import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import pygame

Color = Tuple[int, int, int]

WIDTH = 1200
HEIGHT = 760
FPS = 60
CANVAS_RECT = pygame.Rect(40, 110, WIDTH - 80, HEIGHT - 140)
HEADER_RECT = pygame.Rect(28, 18, WIDTH - 56, 78)

STEP_INTERVAL = 0.22
MAX_ITERATIONS = 30

BG_TOP = (11, 17, 29)
BG_BOTTOM = (3, 7, 14)
TEXT_PRIMARY = (234, 242, 255)
TEXT_MUTED = (142, 160, 188)

POINT_BASE_COLORS: Sequence[Color] = (
    (255, 190, 112),
    (136, 226, 255),
    (255, 154, 194),
    (170, 240, 182),
    (198, 175, 255),
)

CLUSTER_COLORS: Sequence[Color] = (
    (70, 224, 255),
    (255, 122, 138),
    (252, 214, 102),
    (136, 228, 156),
    (180, 150, 255),
    (255, 160, 104),
    (113, 191, 255),
    (255, 125, 211),
)


@dataclass
class Point:
    x: float
    y: float
    base_color: Color
    draw_color: Color
    target_color: Color
    cluster: int = -1
    pulse_seed: float = 0.0

    def update(self) -> None:
        self.draw_color = lerp_color(self.draw_color, self.target_color, 0.16)


@dataclass
class Centroid:
    x: float
    y: float
    target_x: float
    target_y: float
    color: Color

    def update(self) -> None:
        self.x = lerp(self.x, self.target_x, 0.22)
        self.y = lerp(self.y, self.target_y, 0.22)


@dataclass
class Button:
    rect: pygame.Rect
    label: str
    color: Color


def clamp_channel(value: float) -> int:
    return max(0, min(255, int(value)))


def lerp(start: float, end: float, amount: float) -> float:
    return start + (end - start) * amount


def lerp_color(start: Color, end: Color, amount: float) -> Color:
    return (
        clamp_channel(lerp(start[0], end[0], amount)),
        clamp_channel(lerp(start[1], end[1], amount)),
        clamp_channel(lerp(start[2], end[2], amount)),
    )


def tint(color: Color, multiplier: float) -> Color:
    return (
        clamp_channel(color[0] * multiplier),
        clamp_channel(color[1] * multiplier),
        clamp_channel(color[2] * multiplier),
    )


def load_font(size: int, bold: bool = False) -> pygame.font.Font:
    for family in ("montserrat", "poppins", "segoeui", "verdana"):
        match = pygame.font.match_font(family, bold=bold)
        if match:
            return pygame.font.Font(match, size)
    return pygame.font.Font(None, size)


def make_background(width: int, height: int) -> pygame.Surface:
    background = pygame.Surface((width, height))
    for y in range(height):
        amount = y / max(1, height - 1)
        pygame.draw.line(
            background,
            lerp_color(BG_TOP, BG_BOTTOM, amount),
            (0, y),
            (width, y),
        )

    atmosphere = pygame.Surface((width, height), pygame.SRCALPHA)
    circles = (
        (210, 180, 260, (60, 120, 190, 60)),
        (980, 140, 280, (54, 94, 180, 55)),
        (760, 600, 340, (70, 120, 150, 38)),
        (260, 640, 280, (86, 68, 150, 26)),
    )
    for cx, cy, radius, color in circles:
        pygame.draw.circle(atmosphere, color, (cx, cy), radius)
    background.blit(atmosphere, (0, 0))
    return background


def make_grid_layer(width: int, height: int) -> pygame.Surface:
    grid = pygame.Surface((width, height), pygame.SRCALPHA)
    for x in range(0, width, 36):
        pygame.draw.line(grid, (118, 141, 173, 28), (x, 0), (x, height), 1)
    for y in range(0, height, 36):
        pygame.draw.line(grid, (118, 141, 173, 28), (0, y), (width, y), 1)
    return grid


def draw_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    color: Color,
    alpha: int,
    radius: int,
) -> None:
    panel = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(panel, (*color, alpha), panel.get_rect(), border_radius=radius)
    surface.blit(panel, rect.topleft)


def draw_button(
    surface: pygame.Surface,
    button: Button,
    font: pygame.font.Font,
    hovered: bool,
    enabled: bool = True,
) -> None:
    base_color = button.color if enabled else (74, 85, 104)
    fill = tint(base_color, 1.12 if hovered and enabled else 1.0)
    draw_panel(surface, button.rect, fill, 220 if enabled else 130, 14)
    pygame.draw.rect(surface, tint(fill, 1.3), button.rect, 2, border_radius=14)

    label_color = (12, 17, 29) if enabled else (170, 180, 198)
    label = font.render(button.label, True, label_color)
    surface.blit(label, label.get_rect(center=button.rect.center))


def draw_centroid(surface: pygame.Surface, centroid: Centroid, runtime: float) -> None:
    center_x = int(centroid.x)
    center_y = int(centroid.y)
    pulse = 21 + int(3 * math.sin(runtime * 4.0 + center_x * 0.01))

    glow = pygame.Surface((140, 140), pygame.SRCALPHA)
    pygame.draw.circle(glow, (*centroid.color, 68), (70, 70), pulse)
    pygame.draw.circle(glow, (*centroid.color, 120), (70, 70), 13)
    surface.blit(glow, (center_x - 70, center_y - 70))

    pygame.draw.circle(surface, centroid.color, (center_x, center_y), 9)
    pygame.draw.circle(surface, (245, 248, 255), (center_x, center_y), 15, 2)
    pygame.draw.line(surface, (245, 248, 255), (center_x - 17, center_y), (center_x + 17, center_y), 2)
    pygame.draw.line(surface, (245, 248, 255), (center_x, center_y - 17), (center_x, center_y + 17), 2)


def add_point(points: List[Point], x_pos: float, y_pos: float) -> None:
    base_color = random.choice(POINT_BASE_COLORS)
    points.append(
        Point(
            x=x_pos,
            y=y_pos,
            base_color=base_color,
            draw_color=base_color,
            target_color=base_color,
            pulse_seed=random.uniform(0.0, math.tau),
        )
    )


def add_random_points(points: List[Point], count: int) -> None:
    padding = 22
    for _ in range(count):
        x_pos = random.uniform(CANVAS_RECT.left + padding, CANVAS_RECT.right - padding)
        y_pos = random.uniform(CANVAS_RECT.top + padding, CANVAS_RECT.bottom - padding)
        add_point(points, x_pos, y_pos)


def clear_assignments(points: List[Point], centroids: List[Centroid]) -> None:
    centroids.clear()
    for point in points:
        point.cluster = -1
        point.target_color = point.base_color


def initialize_centroids(points: Sequence[Point], cluster_count: int) -> List[Centroid]:
    selected = random.sample(list(points), cluster_count)
    centroids: List[Centroid] = []
    for index, point in enumerate(selected):
        color = CLUSTER_COLORS[index % len(CLUSTER_COLORS)]
        centroids.append(Centroid(point.x, point.y, point.x, point.y, color))
    return centroids


def run_kmeans_iteration(points: Sequence[Point], centroids: Sequence[Centroid]) -> Tuple[bool, float]:
    changed = False

    for point in points:
        closest_index = 0
        closest_distance = float("inf")
        for index, centroid in enumerate(centroids):
            dx = point.x - centroid.x
            dy = point.y - centroid.y
            distance = dx * dx + dy * dy
            if distance < closest_distance:
                closest_distance = distance
                closest_index = index
        if point.cluster != closest_index:
            changed = True
        point.cluster = closest_index
        point.target_color = centroids[closest_index].color

    shift_total = 0.0
    for index, centroid in enumerate(centroids):
        members = [point for point in points if point.cluster == index]
        if members:
            target_x = sum(point.x for point in members) / len(members)
            target_y = sum(point.y for point in members) / len(members)
        else:
            fallback = random.choice(points)
            target_x = fallback.x
            target_y = fallback.y

        shift_total += math.hypot(centroid.target_x - target_x, centroid.target_y - target_y)
        centroid.target_x = target_x
        centroid.target_y = target_y

    return changed, shift_total


def main() -> None:
    pygame.init()
    pygame.display.set_caption("K-Means Playground")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    title_font = load_font(34, bold=True)
    body_font = load_font(22)
    small_font = load_font(18)

    background = make_background(WIDTH, HEIGHT)
    grid_layer = make_grid_layer(CANVAS_RECT.width, CANVAS_RECT.height)

    start_button = Button(pygame.Rect(44, 32, 138, 50), "Start", (74, 224, 164))
    reset_button = Button(pygame.Rect(196, 32, 138, 50), "Reset", (255, 118, 128))
    scatter_button = Button(pygame.Rect(348, 32, 138, 50), "Scatter", (116, 173, 255))

    points: List[Point] = []
    centroids: List[Centroid] = []
    cluster_count = 4
    clustering = False
    iteration = 0
    elapsed_step = 0.0
    runtime = 0.0
    status_text = "Click in the canvas to add points. Press Start or Space."

    def stop_and_clear_assignments(new_status: str) -> None:
        nonlocal clustering, iteration, elapsed_step, status_text
        clustering = False
        iteration = 0
        elapsed_step = 0.0
        clear_assignments(points, centroids)
        status_text = new_status

    def start_clustering() -> None:
        nonlocal centroids, clustering, iteration, elapsed_step, status_text
        if len(points) < 2:
            status_text = "Add at least 2 points before clustering."
            return

        active_cluster_count = max(2, min(cluster_count, len(points)))
        centroids = initialize_centroids(points, active_cluster_count)
        clustering = True
        iteration = 0
        elapsed_step = STEP_INTERVAL
        status_text = f"Clustering {len(points)} points into {active_cluster_count} groups."

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        runtime += dt
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    start_clustering()
                elif event.key == pygame.K_r:
                    points.clear()
                    stop_and_clear_assignments("Canvas cleared. Click to add new points.")
                elif event.key in (pygame.K_LEFTBRACKET, pygame.K_MINUS):
                    cluster_count = max(2, cluster_count - 1)
                    status_text = f"Cluster count set to {cluster_count}."
                elif event.key in (pygame.K_RIGHTBRACKET, pygame.K_EQUALS, pygame.K_PLUS):
                    cluster_count = min(8, cluster_count + 1)
                    status_text = f"Cluster count set to {cluster_count}."

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if start_button.rect.collidepoint(event.pos):
                    start_clustering()
                elif reset_button.rect.collidepoint(event.pos):
                    points.clear()
                    stop_and_clear_assignments("Canvas cleared. Click to add new points.")
                elif scatter_button.rect.collidepoint(event.pos):
                    add_random_points(points, 45)
                    stop_and_clear_assignments("Random points generated. Press Start to cluster.")
                elif CANVAS_RECT.collidepoint(event.pos):
                    add_point(points, float(event.pos[0]), float(event.pos[1]))
                    stop_and_clear_assignments(f"Added point {len(points)}. Press Start to cluster.")

        if clustering and centroids:
            elapsed_step += dt
            if elapsed_step >= STEP_INTERVAL:
                elapsed_step = 0.0
                changed, shift = run_kmeans_iteration(points, centroids)
                iteration += 1
                if (not changed and shift < 0.6) or iteration >= MAX_ITERATIONS:
                    clustering = False
                    status_text = f"Done in {iteration} iterations. Add more points or press Start again."

        for point in points:
            point.update()
        for centroid in centroids:
            centroid.update()

        screen.blit(background, (0, 0))
        draw_panel(screen, HEADER_RECT, (15, 22, 36), 220, 24)
        draw_panel(screen, CANVAS_RECT, (10, 15, 26), 212, 22)
        pygame.draw.rect(screen, (86, 108, 142), CANVAS_RECT, 2, border_radius=22)
        screen.blit(grid_layer, CANVAS_RECT.topleft)

        link_layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        line_alpha = 96 if clustering else 42
        for point in points:
            if 0 <= point.cluster < len(centroids):
                centroid = centroids[point.cluster]
                pygame.draw.line(
                    link_layer,
                    (*centroid.color, line_alpha),
                    (int(point.x), int(point.y)),
                    (int(centroid.x), int(centroid.y)),
                    1,
                )
        screen.blit(link_layer, (0, 0))

        point_layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for point in points:
            px = int(point.x)
            py = int(point.y)
            glow_radius = 7 + int(2 * math.sin(runtime * 4.2 + point.pulse_seed))
            pygame.draw.circle(point_layer, (*point.draw_color, 72), (px, py), glow_radius)
            pygame.draw.circle(point_layer, point.draw_color, (px, py), 4)
        screen.blit(point_layer, (0, 0))

        for centroid in centroids:
            draw_centroid(screen, centroid, runtime)

        draw_button(
            screen,
            start_button,
            body_font,
            hovered=start_button.rect.collidepoint(mouse_pos),
            enabled=len(points) >= 2,
        )
        draw_button(
            screen,
            reset_button,
            body_font,
            hovered=reset_button.rect.collidepoint(mouse_pos),
            enabled=bool(points),
        )
        draw_button(
            screen,
            scatter_button,
            body_font,
            hovered=scatter_button.rect.collidepoint(mouse_pos),
            enabled=True,
        )

        title_surface = title_font.render("K-Means Playground", True, TEXT_PRIMARY)
        screen.blit(title_surface, (510, 28))

        hint_surface = small_font.render(
            "Click canvas: add points   |   [ and ] change K   |   Space starts",
            True,
            TEXT_MUTED,
        )
        screen.blit(hint_surface, (510, 63))

        active_k = max(1, min(cluster_count, len(points))) if points else cluster_count
        stats_surface = small_font.render(
            f"Points: {len(points)}    K: {active_k}    Iteration: {iteration}",
            True,
            (156, 233, 197) if clustering else TEXT_MUTED,
        )
        screen.blit(stats_surface, (CANVAS_RECT.left + 18, CANVAS_RECT.top + 16))

        status_surface = body_font.render(status_text, True, TEXT_PRIMARY if clustering else TEXT_MUTED)
        screen.blit(status_surface, (CANVAS_RECT.left + 18, CANVAS_RECT.bottom - 34))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
