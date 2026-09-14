"""Thresholds shared by the reporting scripts. Both filters are load-bearing:
MIN_VOL screens on the projection (kills backups who will not play), BETTABLE
screens on the baseline (kills players with no market)."""
MIN_VOL = {"passing_yards": 20.0, "completions": 20.0, "rushing_yards": 7.0,
           "receiving_yards": 3.0, "receptions": 3.0}
BETTABLE = {"passing_yards": 150.0, "completions": 12.0, "rushing_yards": 25.0,
            "receiving_yards": 25.0, "receptions": 2.0}
