from __future__ import annotations

import random

from trig_math import PI, TrigFunction


def pick_random_trig(
    include_sin: bool,
    include_cos: bool,
    include_tan: bool,
    include_cot: bool,
    include_sec: bool,
    include_csc: bool,
) -> str:
    choices = []
    if include_sin:
        choices.append("sin")
    if include_cos:
        choices.append("cos")
    if include_tan:
        choices.append("tan")
    if include_cot:
        choices.append("cot")
    if include_sec:
        choices.append("sec")
    if include_csc:
        choices.append("csc")
    if not choices:
        return "sin"
    return random.choice(choices)


def random_param_set(rand_A: bool, rand_B: bool, rand_C: bool, rand_D: bool) -> TrigFunction:
    # Base (non-randomized) defaults
    A = 1.0
    B = 1.0
    C = 0.0
    D = 0.0

    # Amplitude (include sign flip only when randomized)
    if rand_A:
        A = float(random.choice([1, 2, 3, 4, 5]) * random.choice([1, -1]))

    # Frequency B (period/stretches). Keep positive by default (class-friendly).
    if rand_B:
        B = float(random.choice([0.5, 1, 2, 3]))

    # Phase shift C (horizontal shift)
    if rand_C:
        C = float(random.choice([k * (PI / 6) for k in range(-12, 13)]))

    # Vertical shift D
    if rand_D:
        D = float(random.choice([-3, -2, -1, 0, 1, 2, 3]))

    return TrigFunction(kind="sin", A=A, B=B, C=C, D=D)


def generate_question(
    include_sin: bool,
    include_cos: bool,
    include_tan: bool,
    include_cot: bool,
    include_sec: bool,
    include_csc: bool,
    rand_A: bool,
    rand_B: bool,
    rand_C: bool,
    rand_D: bool,
) -> TrigFunction:
    tf = random_param_set(rand_A, rand_B, rand_C, rand_D)
    tf = TrigFunction(
        kind=pick_random_trig(include_sin, include_cos, include_tan, include_cot, include_sec, include_csc),
        A=tf.A,
        B=tf.B,
        C=tf.C,
        D=tf.D,
    )
    return tf
