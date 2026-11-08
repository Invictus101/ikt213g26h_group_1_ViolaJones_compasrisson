import cv2

from TrackedFace import TrackedFace


def put_text_outlined(frame, text, org, font, scale, color, thickness, outline_color=(0, 0, 0), outline_radius=1):
    """draws text with an outline so its easier to read"""
    x, y = org
    for dx in range(-outline_radius, outline_radius + 1):
        for dy in range(-outline_radius, outline_radius + 1):
            if (dx, dy) != (0, 0) and dx * dx + dy * dy <= outline_radius * outline_radius + 1:
                cv2.putText(frame, text, (x + dx, y + dy), font, scale, outline_color, thickness, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)


def score_color(value):
    """BGR color for a 0-1 value: green >= 0.75, yellow >= 0.5, orange >= 0.25, else red"""
    if value >= 0.75:
        return (0, 255, 0)
    if value >= 0.5:
        return (0, 255, 255)
    if value >= 0.25:
        return (0, 165, 255)
    return (0, 0, 255)


def label_text_color(background):
    """black text on bright backgrounds (yellow), white otherwise"""
    b, g, r = background
    brightness = 0.114 * b + 0.587 * g + 0.299 * r # perceived brightness, eyes are most sensitive to green
    return (0, 0, 0) if brightness > 200 else (255, 255, 255)


def draw_facebox(frame, face_boxes, label, color, stats=None, thickness=2):
    """Draws a colored box around the face, with the name of guessed person and other small colored stats above it.
    stats: list<(text, color)>. thickness: line width of the box"""

    x0, y0, x1, y1 = map(int, face_boxes[:4])

    # Draws the rectangle around the face
    cv2.rectangle(frame, (x0, y0), (x0 + x1, y0 + y1), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    # stats text (smaller, colored) goes above the label
    stats_font_scale = 0.45
    stats_thickness = 1
    stats_gap = 10 # px between each stat
    stats_space = 0
    if stats:
        (_, stats_height), _ = cv2.getTextSize(stats[0][0], font, stats_font_scale, stats_thickness)
        stats_space = stats_height + 6

    # push label down if needed, so top text still gets shown if the frame is partly outside the top part of screen
    text_y = max(y0 - 10, text_height + 10 + stats_space)

    if stats:
        stat_x = x0 + 3
        stat_y = text_y - text_height - 8
        for stat_text, stat_color in stats:
            put_text_outlined(frame, stat_text, (stat_x, stat_y), font, stats_font_scale, stat_color, stats_thickness)
            (stat_width, _), _ = cv2.getTextSize(stat_text, font, stats_font_scale, stats_thickness)
            stat_x += stat_width + stats_gap

    cv2.rectangle(
        frame,
        (x0, text_y - text_height - 4),
        (x0 + text_width + 6, text_y + baseline),
        color,
        -1
    )
    cv2.putText(
        frame,
        label,
        (x0 + 3, text_y - 2),
        font,
        font_scale,
        label_text_color(color),
        font_thickness,
        cv2.LINE_AA
    )
    return frame


def draw_tracked_faces(frame, tracked_faces: list[TrackedFace]):
    """Draws a debug box for every tracked face. box + label colored by confidence, small stats colored by their own factor"""
    for face in tracked_faces:
        name, similarity = face.best_match
        confidence = face.confidence # if 2 persons have almost the same score, confidence will be low
        color = score_color(confidence)

        # red -> Unknown, yellow/orange -> "Name?", green -> "Name"
        if confidence < 0.25:
            name = "Unknown"
        elif confidence < 0.75:
            name += "?"

        stats = [
            (f"similar {similarity:.2f}", score_color(face.similarity_base)),
            (f"certain {face.certainty:.2f}", score_color(face.certainty_strength)),
            (f"{face.spoofing["label"]}: {face.spoofing["real_score"]:.2f}", score_color(face.spoofing["real_score"])),
        ]
        if face.is_missing:
            stats.append((f"last seen {face.seconds_missing:.1f}s", score_color(face.time_factor)))

        draw_facebox_debug( # also shows ppl that almost matched
            frame         = frame,
            face_boxes    = face.box,
            label         = f"{name} ({confidence:.2f})",
            color         = color,
            stats         = stats,
            person_scores = face.scores,
            thickness     = 1 if face.is_missing else 2 # thinner box when not detected this frame
        )
    return frame


def draw_facebox_debug(frame, face_boxes, label, color, stats=None, person_scores=None, thickness=2):
    """Same as draw_facebox, but also lists every candidate person and score on the right side of the box"""
    draw_facebox(frame, face_boxes, label, color, stats, thickness)

    if not person_scores:
        return frame

    x, y, w, h = map(int, face_boxes[:4])

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_thickness = 1
    line_height = 16

    text_x = x + w + 6
    for i, (person_name, score) in enumerate(person_scores.items()):
        text_y = y + 12 + i * line_height
        line = f"{person_name}: {score:.2f}"

        put_text_outlined(frame, line, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    return frame