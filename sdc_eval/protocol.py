"""Which month each evaluation episode is drawn from: shared by eval_compare.py and sdc_lp_bound.py.

Month specs (the env config's `month`, or --month on the command line):
    0..11   every episode starts within +-7 days of that month's first day (the env's own rule)
    all     every episode starts on a random day of the year (SustainDC, month 'all'); used for training
    cycle   evaluation only: episode i uses month i % 12 (seed base + i), so 12 episodes cover each month once.
            Each episode is then an ordinary fixed-month episode, so its seed alone fixes it, as for any month.
"""
MONTH_KEYWORDS = ('all', 'cycle')


def parse_month(value):
    """'6' / 6 -> 6, 'all' / 'cycle' -> the keyword, None -> None. Raises ValueError otherwise."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in MONTH_KEYWORDS:
        return value.strip().lower()
    month = int(value)
    if not 0 <= month <= 11:
        raise ValueError(f'month must be 0..11, all or cycle, got {value!r}')
    return month


def episode_months(month, n):
    """The env-config month of each of n episodes: i % 12 for 'cycle', otherwise `month` for every episode."""
    if month == 'cycle':
        return [i % 12 for i in range(n)]
    return [month] * n
