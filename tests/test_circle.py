from spyral.geometry.circle import generate_circle_points, least_squares_circle


def test_circle():

    center_x = 0.0
    center_y = 0.0
    radius = 1.0

    points = generate_circle_points(center_x, center_y, radius)
    cx, cy, r, _ = least_squares_circle(points[:, 0], points[:, 1])

    assert abs(cx - center_x) < 0.1
    assert abs(cy - center_y) < 0.1
    assert abs(r - radius) < 0.1
