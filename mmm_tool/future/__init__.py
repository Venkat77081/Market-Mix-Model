"""Future planning and tracking (Phases G and H in Future mode, and the Tracking screen).

  period.py    the plan's weeks, the template weeks it copies, the lead-in weeks (no Meridian)
  forecast.py  "the KPI if spend stays like last year": built in, a CSV, one total, or none
  data.py      build_future_data: Meridian's DataTensors for the lead-in + plan weeks
  engine.py    media outcome per plan week on every posterior draw, quick curves, the optimizer
  store.py     the project's plans, the tracked plan, tracking settings and the planning log
  tracking.py  actuals against the tracked plan: spend vs plan, the KPI against its prediction
  __main__.py  the background jobs (Meridian runs there, never in the web server)

Everything is read from the model, its table and the project's settings: nothing here names a
channel, a column, a date, a currency or a KPI.
"""
