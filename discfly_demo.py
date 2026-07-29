# AI Disc Golf Physics Lab (DiscFly-AI)
Final project for the Building AI course

## Summary
The AI Disc Golf Physics Lab is an interactive teaching aid designed for high school and university physics students. By applying Newtonian mechanics, specifically aerodynamics, lift, drag, rotational torque, and gyroscopic precession, students use a machine learning model to predict and analyze disc golf flight paths based on release parameters (speed, spin, release angle, nose angle, and wind).

## Background
Physics concepts like aerodynamics, torque, and rotational dynamics can feel abstract in traditional textbook exercises. Disc golf provides a tangible, real-world laboratory where classic physics laws directly dictate visible, dramatic outcomes.

Key educational challenges this project solves:
- **Bridging Theory and Intuition:** Connects kinematic equations ($F=ma$, Bernoulli's principle, angular momentum $L=I\omega$) to real-world projectile motion across four distinct flight phases.
- **Understanding Non-Ideal Dynamics:** Standard physics textbook problems ignore air resistance. Disc golf requires modeling drag, aerodynamic lift, and gyroscopic stability (fade/turn).
- **AI in Physics:** Teaches students how machine learning can augment physical models by predicting complex turbulent air interactions that are difficult to solve analytically.

This project directly builds upon my published research on using disc golf dynamics to break down and teach multi-phase aerodynamic flight paths to physics students.

## Published Research & Prior Work
This project is grounded in my peer-reviewed educational research:
- **Title:** *Using Disc Golf to Teach the Four Phases of Projectile Flight*
- **Journal:** *Physics Education* (*Phys. Educ.* Vol. 61, 043002, June 2026)
- **DOI:** [10.1088/1361-6552/ae7790](https://doi.org/10.1088/1361-6552/ae7790)

The four phases established in this paper, **Launch/High-Speed Turn**, **High-Speed Stability**, **Low-Speed Fade**, and **Ground Impact/Slide**, serve as the foundational physical constraints for our machine learning trajectory framework.

## How is it used?
The project serves as an open-source educational module for physics classrooms, outdoor field labs, and self-directed learning.

1. **Input Stage:** Students measure or input release variables: release velocity ($v$), spin rate ($\omega$), launch angle ($\theta$), nose angle ($\alpha$), and disc mass/properties.
2. **Physics Calculation:** A baseline kinematic script projects the trajectory across the flight phases using standard gravitational and basic aerodynamic force vectors (Lift $F_L$, Drag $F_D$).
3. **AI Prediction:** A trained regression model adjusts the trajectory to account for phase transitions (high-speed turn to low-speed fade) and wind vectors.
4. **Visual Analysis:** Students compare predicted vs. actual throw distances/paths and analyze the physics forces driving discrepancies.

The script below demonstrates the full pipeline: a physics baseline, a four-phase flight simulator standing in for real flight-tracking data, and a regression model trained to predict landing coordinates from release parameters. Requires `numpy` and `scikit-learn`.

```python
"""
AI Disc Golf Physics Lab (DiscFly-AI)
Demonstrates the full pipeline described in the README:
  1. Physics-based aerodynamic force calculation
  2. A synthetic flight simulator spanning the four flight phases
     (Launch/High-Speed Turn, High-Speed Stability, Low-Speed Fade, Ground Impact)
  3. A regression model trained on simulated flight data to predict
     landing coordinates from release parameters, with turbulence noise
     standing in for real-world wind gust variance.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error


# --- 1. Core aerodynamic force calculation --------------------------------

def calculate_aerodynamic_forces(velocity, air_density, disc_area, C_lift, C_drag):
    """Calculates aerodynamic lift and drag on a disc golf disc."""
    dynamic_pressure = 0.5 * air_density * (velocity ** 2)
    lift_force = C_lift * dynamic_pressure * disc_area
    drag_force = C_drag * dynamic_pressure * disc_area
    return lift_force, drag_force


# --- 2. Simplified four-phase flight simulator -----------------------------
# This is a lightweight kinematic approximation, not a full CFD model. It
# generates physically-motivated training data for the regression model
# below, standing in for the video-tracked flight data described in the
# README's Data Sources section.

def simulate_flight(velocity, spin_rpm, launch_angle_deg, nose_angle_deg,
                     wind_mps, turn_rating=-1.0, fade_rating=2.0,
                     turbulence_std=0.4, rng=None):
    """
    Simulates a disc golf flight and returns landing coordinates (x, y).

    x: distance downrange (m)
    y: lateral drift from the throwing line (m), positive = turn direction

    Phase 1 (Launch/High-Speed Turn): high speed + high spin resists torque,
      disc turns opposite its spin direction proportional to turn_rating.
    Phase 2 (High-Speed Stability): disc holds a fairly straight line.
    Phase 3 (Low-Speed Fade): as spin/speed decay, gyroscopic stability
      drops and the disc fades back in its spin direction (fade_rating).
    Phase 4 (Ground Impact/Slide): small additional drift/slide on landing.
    """
    if rng is None:
        rng = np.random.default_rng()

    launch_rad = np.radians(launch_angle_deg)
    nose_rad = np.radians(nose_angle_deg)

    # Distance: driven by release speed and launch angle, penalized by
    # excess nose-up angle (adds drag) and rewarded by spin (stability).
    base_distance = 0.9 * velocity ** 1.5 * np.sin(2 * launch_rad) ** 0.5
    nose_penalty = 1.0 - 0.4 * max(nose_angle_deg, 0) / 30.0
    spin_bonus = 1.0 + 0.05 * (spin_rpm / 1000.0)
    distance = max(base_distance * nose_penalty * spin_bonus, 1.0)

    # Lateral drift: turn phase scales with speed, fade phase scales with
    # distance traveled (spin/speed decay), wind adds a linear drift term.
    turn_drift = turn_rating * (velocity / 25.0) * 3.0
    fade_drift = fade_rating * (distance / 100.0) * 4.0
    wind_drift = wind_mps * (distance / 20.0)

    lateral = turn_drift + fade_drift + wind_drift

    # Turbulence / measurement noise (Air Turbulence + Data Collection
    # Precision challenges noted in the README).
    distance += rng.normal(0, turbulence_std * 2.0)
    lateral += rng.normal(0, turbulence_std)

    return distance, lateral


# --- 3. Generate a synthetic training set -----------------------------------

def build_dataset(n_samples=2000, seed=42):
    rng = np.random.default_rng(seed)

    velocity = rng.uniform(15, 27, n_samples)          # m/s
    spin_rpm = rng.uniform(400, 1200, n_samples)        # rpm
    launch_angle = rng.uniform(5, 20, n_samples)        # degrees
    nose_angle = rng.uniform(-5, 10, n_samples)         # degrees
    wind = rng.uniform(-4, 4, n_samples)                # m/s, +downwind

    X = np.column_stack([velocity, spin_rpm, launch_angle, nose_angle, wind])
    y = np.array([
        simulate_flight(v, s, la, na, w, rng=rng)
        for v, s, la, na, w in X
    ])
    return X, y


# --- 4. Train the AI prediction model --------------------------------------

def train_model():
    X, y = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        max_iter=2000,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    return model, mae


if __name__ == "__main__":
    # Baseline physics calculation (as in the original snippet)
    lift, drag = calculate_aerodynamic_forces(
        velocity=25, air_density=1.225, disc_area=0.035, C_lift=0.3, C_drag=0.15
    )
    print(f"Lift Force: {lift:.2f} N | Drag Force: {drag:.2f} N\n")

    # Train the AI trajectory model and report accuracy
    model, mae = train_model()
    print(f"Trained landing-coordinate model. Mean absolute error: {mae:.2f} m\n")

    # Example: predict a landing spot for a specific throw and compare
    # against a fresh physics simulation of the same release parameters
    test_throw = np.array([[25, 900, 12, 2, 1.5]])  # v, spin, launch, nose, wind
    predicted_landing = model.predict(test_throw)[0]
    actual_landing = simulate_flight(25, 900, 12, 2, 1.5, rng=np.random.default_rng(7))

    print(f"AI-predicted landing (distance, lateral): "
          f"({predicted_landing[0]:.1f} m, {predicted_landing[1]:.1f} m)")
    print(f"Physics-simulated landing (distance, lateral): "
          f"({actual_landing[0]:.1f} m, {actual_landing[1]:.1f} m)")

```

## Data sources and AI methods

**Data Sources:**
- Empirical flight data and phase transition boundaries defined in Phys. Educ. 61 043002.
- Open flight chart datasets from disc manufacturers (PDGA specifications, speed/glide/turn/fade ratings).
- Video trajectory tracking data collected during field physics labs.

**Physics Integration:**
- Newton's 2nd Law ($F = ma$) in 3D space across all four flight phases.
- Aerodynamic lift and drag ($F_L = \frac{1}{2}\rho v^2 A C_L$, $F_D = \frac{1}{2}\rho v^2 A C_D$).
- Angular momentum and gyroscopic precession ($\tau = \frac{dL}{dt}$).

**AI/ML Methods:**
- Multivariate regression / neural networks: predicting final landing coordinates ($x, y$) and phase-change points based on flight ratings and initial release metrics.

## Challenges
- **Air Turbulence:** Real-world wind gusts create non-deterministic flight behaviors that ideal physics models cannot fully anticipate.
- **Data Collection Precision:** Accurately measuring spin rate ($\text{RPM}$) and nose angle without high-speed cameras or specialized sensors can be difficult for standard classrooms.

## What next?
- **Interactive Web App:** Build a browser-based simulator (using Streamlit or Plotly) where students can move sliders for release speed/spin and immediately see 3D flight paths across all four phases.
- **Sensor Integration:** Create a smart disc or smartphone app integration using computer vision to auto-extract release velocity and launch angle from video.
- **Classroom Lesson Plans:** Expand on the lesson guides from Phys. Educ. 61 043002 into free downloadable digital lab worksheets for standard physics curricula.

## Acknowledgments
- Building AI course by Reaktor Innovations and the University of Helsinki.
- IOP Publishing (Physics Education journal).
- Professional Disc Golf Association (PDGA) technical standards.
- Concepts based on classical aerodynamics and rigid body dynamics.
