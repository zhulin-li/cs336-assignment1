import time


class Timer:
    """A context manager to time a block of code."""

    def __init__(self, text="Elapsed time: {:.4f} seconds"):
        """
        Initialize the Timer.

        Args:
            text (str): The text to display when printing the elapsed time.
                        It should include a format specifier for the time.
        """
        self._start_time = None
        self.text = text

    def __enter__(self):
        """Start the timer when entering the 'with' block."""
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stop the timer and print the elapsed time when exiting the 'with' block.
        """
        elapsed_time = time.perf_counter() - self._start_time
        print(self.text.format(elapsed_time))
        # Return False to propagate any exceptions that occurred inside the 'with' block.
        return False
