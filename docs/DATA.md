# Data tables

Generated from `cfbmodel/schema.py` (a test keeps this file in step). Columns CFBD sends that are not listed here are passed through unchanged.

## `teams_fbs`

One row per FBS team for a season (/teams/fbs). The source of canonical names.

| column | meaning of the type | required |
|---|---|---|
| `school` | text, never blank | yes |
| `conference` | text, blank allowed | no |
| `classification` | text, blank allowed | no |

No two rows may share the same (school).

Fields CFBD must send: `school`.

## `games`

One row per game (/games), regular season and postseason.

| column | meaning of the type | required |
|---|---|---|
| `id` | whole number, never blank | yes |
| `season` | whole number, never blank | yes |
| `week` | whole number, blank allowed | yes |
| `start_date` | UTC timestamp | yes |
| `homeTeam` | text, never blank | yes |
| `awayTeam` | text, never blank | yes |
| `neutralSite` | True/False | yes |
| `homePoints` | number, blank allowed | yes |
| `awayPoints` | number, blank allowed | yes |
| `margin` | number, blank allowed | yes |
| `total` | number, blank allowed | yes |
| `home_is_fbs` | True/False | yes |
| `away_is_fbs` | True/False | yes |
| `season_type` | text, blank allowed | no |
| `homeConference` | text, blank allowed | no |
| `awayConference` | text, blank allowed | no |
| `venue` | text, blank allowed | no |

No two rows may share the same (id).

Fields CFBD must send: `id`, `season`, `week`, `homeTeam`, `awayTeam`, `startDate`.

## `lines`

One row per game per sportsbook (/lines): opening and closing numbers.

| column | meaning of the type | required |
|---|---|---|
| `gameId` | whole number, never blank | yes |
| `season` | whole number, blank allowed | yes |
| `week` | whole number, blank allowed | yes |
| `home` | text, never blank | yes |
| `away` | text, never blank | yes |
| `book` | text, never blank | yes |
| `spread_close` | number, blank allowed | yes |
| `spread_open` | number, blank allowed | yes |
| `total_close` | number, blank allowed | yes |
| `total_open` | number, blank allowed | yes |
| `ml_home` | number, blank allowed | yes |
| `ml_away` | number, blank allowed | yes |
| `home_is_fbs` | True/False | yes |
| `away_is_fbs` | True/False | yes |
| `formatted_spread` | text, blank allowed | no |
| `homePoints` | number, blank allowed | no |
| `awayPoints` | number, blank allowed | no |

No two rows may share the same (gameId, book).

Fields CFBD must send: `id`, `season`, `week`, `homeTeam`, `awayTeam`, `lines`, `lines[].provider`.

## `plays`

One row per play (/plays), fetched one week at a time, FBS classification.

| column | meaning of the type | required |
|---|---|---|
| `season` | whole number, never blank | yes |
| `week` | whole number, never blank | yes |
| `offense` | text, never blank | yes |
| `defense` | text, never blank | yes |
| `offense_is_fbs` | True/False | yes |
| `defense_is_fbs` | True/False | yes |
| `gameId` | whole number, never blank | no |
| `playType` | text, blank allowed | no |
| `ppa` | number, blank allowed | no |
| `period` | number, blank allowed | no |
| `offenseScore` | number, blank allowed | no |
| `defenseScore` | number, blank allowed | no |
| `home` | text, blank allowed | no |
| `away` | text, blank allowed | no |
| `offenseConference` | text, blank allowed | no |
| `defenseConference` | text, blank allowed | no |

Fields CFBD must send: `gameId`, `offense`, `defense`, `playType`, `ppa`, `period`.

## `player_box`

One row per player per stat per game (/games/players), one week at a time.

| column | meaning of the type | required |
|---|---|---|
| `gameId` | whole number, never blank | yes |
| `season` | whole number, never blank | yes |
| `week` | whole number, never blank | yes |
| `team` | text, never blank | yes |
| `category` | text, never blank | yes |
| `stat_type` | text, never blank | yes |
| `player` | text, never blank | yes |
| `team_is_fbs` | True/False | yes |
| `athlete_id` | text, blank allowed | no |
| `value` | text, blank allowed | no |
| `conference` | text, blank allowed | no |

Fields CFBD must send: `id`, `teams`, `teams[].team`, `teams[].categories`.

## `usage`

Player usage rates (/player/usage). Passed through as CFBD sends it.

| column | meaning of the type | required |
|---|---|---|
| `team` | text, blank allowed | no |

## `recruiting_teams`

247 composite team recruiting class (/recruiting/teams).

| column | meaning of the type | required |
|---|---|---|
| `team` | text, never blank | yes |
| `points` | number, blank allowed | yes |
| `rank` | number, blank allowed | no |
| `year` | number, blank allowed | no |

Fields CFBD must send: `team`, `points`.

## `returning_production`

Returning production by team (/player/returning).

| column | meaning of the type | required |
|---|---|---|
| `team` | text, never blank | yes |

Fields CFBD must send: `team`.

## `portal`

Transfer portal entries (/player/portal).

| column | meaning of the type | required |
|---|---|---|
| `origin` | text, never blank | yes |
| `destination` | text, blank allowed | no |
| `position` | text, blank allowed | no |
| `rating` | number, blank allowed | no |
| `firstName` | text, blank allowed | no |
| `lastName` | text, blank allowed | no |

Fields CFBD must send: `origin`.

## `venues`

Stadiums (/venues), with the dome flag used to zero weather effects.

| column | meaning of the type | required |
|---|---|---|
| `name` | text, never blank | yes |
| `dome` | True/False/blank (unknown) | no |
| `latitude` | number, blank allowed | no |
| `longitude` | number, blank allowed | no |
| `timezone` | text, blank allowed | no |

Fields CFBD must send: `name`.
