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

```python
import numpy as np

def calculate_aerodynamic_forces(velocity, air_density, disc_area, C_lift, C_drag):
    # Calculates Aerodynamic Lift and Drag on a Disc Golf Disc
    dynamic_pressure = 0.5 * air_density * (velocity ** 2)
    lift_force = C_lift * dynamic_pressure * disc_area
    drag_force = C_drag * dynamic_pressure * disc_area
    return lift_force, drag_force

# Example: Disc thrown at 25 m/s (approx 56 mph)
lift, drag = calculate_aerodynamic_forces(velocity=25, air_density=1.225, disc_area=0.035, C_lift=0.3, C_drag=0.15)
print(f"Lift Force: {lift:.2f} N | Drag Force: {drag:.2f} N")
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
