from time import sleep
from typing import Iterable, Sequence

from gpiozero import OutputDevice


class Stepper28BYJ48:
    """
    Control a 28BYJ-48 stepper motor via a ULN2003 driver board.

    This class implements half-step driving (8-phase sequence).
    """

    #: Half-step coil activation sequence (8 phases)
    # magnetic field configuration
    HALF_STEP_SEQUENCE: Sequence[tuple[int, int, int, int]] = (
        (1, 0, 0, 0),
        (1, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 1, 0),
        (0, 0, 1, 0),
        (0, 0, 1, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
    )

    #: Number of half-steps per full revolution
    STEPS_PER_REVOLUTION: int = 2048 # see specs

    def __init__(
        self,
        pins: Iterable[int],
        delay: float = 0.001,
    ) -> None:
        """
        Initialize the stepper motor controller.

        Args:
            pins:
                An iterable of four BCM GPIO pin numbers connected
                to IN1–IN4 of the ULN2003 driver board.
            delay:
                Delay in seconds between half-steps.
                Smaller values increase speed but reduce torque.
        """
        pins = tuple(pins)
        if len(pins) != 4:
            raise ValueError("Exactly four GPIO pins are required")

        self._coils: list[OutputDevice] = [
            OutputDevice(pin) for pin in pins
        ]
        self.delay: float = delay
        self.position: int = 0  # Half-step position (software-tracked)
        self._phase: int = 0    # Current phase index (0–7)

    # ------------------------------------------------------------------
    # Core motor control
    # ------------------------------------------------------------------

    def step(self, steps: int, direction: int = 1) -> None:
        """
        Move the motor by a given number of half-steps.

        Args:
            steps:
                Number of half-steps to move. Must be non-negative.
            direction:
                Direction of motion:
                +1 for clockwise, -1 for counter-clockwise.

        Raises:
            ValueError: If direction is not +1 or -1.
        """
        if direction not in (1, -1):
            raise ValueError("Direction must be +1 or -1")

        for _ in range(steps):
            self._phase = (self._phase + direction) % 8
            pattern = self.HALF_STEP_SEQUENCE[self._phase]

            for coil, value in zip(self._coils, pattern):
                coil.value = value

            sleep(self.delay)
            self.position += direction

    # ------------------------------------------------------------------
    # Convenience motion helpers
    # ------------------------------------------------------------------

    def rotate_degrees(self, degrees: float) -> None:
        """
        Args:
            degrees: Angle to rotate, in degrees.
        """
        steps = int((degrees / 360.0) * self.STEPS_PER_REVOLUTION)
        direction = 1 if steps >= 0 else -1
        self.step(abs(steps), direction)

    def rotate_revolutions(self, revolutions: float) -> None:
        """

        Args:
            revolutions: Number of full rotations.
        """
        steps = int(revolutions * self.STEPS_PER_REVOLUTION)
        direction = 1 if steps >= 0 else -1
        self.step(abs(steps), direction)

    def set_speed(self, delay: float) -> None:
        """
        Set the delay between half-steps.

        Args:
            delay:
                Delay in seconds. Smaller values increase speed but
                may cause missed steps or stalling.
        """
        self.delay = delay

    # ------------------------------------------------------------------
    # Cleanup / safety
    # ------------------------------------------------------------------

    def release(self) -> None:
        """
        De-energize all motor coils.

        This should be called when the motor is idle to reduce
        heating and power consumption.
        """
        for coil in self._coils:
            coil.off()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "Stepper28BYJ48":
        """Enable use as a context manager."""
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Ensure coils are released on exit."""
        self.release()
