# Exoplanet Data Analysis and Radius Prediction

# Overview 

The project is about two parts. A function that uses public NASA data to calculate approximate transit depths of exoplanets using the radii of exoplanets and their respective host stars. The other part is a machine learning model that tries to predict the exoplanet radii using its temperature, transit depth, orbital period, radius of the star, and another model that uses the same variables but with the addition of the exoplanet's mass to see if the model can accurately predict the radius and how much more accurately. This is mostly a test project to test machine learning models and use it in a real-world case.

## Data 

The data comes from the NASA archive, and my script calls it directly from the archive. The script requests the planet name, observed transit depth, host-star temperature, host-star radius, planet mass, planet radius, and orbital period. There are some measurements missing from some exoplanets; this is due to not every exoplanet having every measurement available.

## Transit Depth Analysis

For the transit depth, both the planet's and star's radii were used to calculate it using:

Transit Depth = (Planet Radius / Star Radius)² × 100

First, the star radius had to be multiplied by 109.2 to convert it to Earth radii so the ratio for the transit depth could be calculated. The NASA transit depth was then used to compare both the calculated and observed transit depths and get their percentage difference.

## Exoplanet Classification and Visualisation

The exoplanets were grouped into radii, as it is one of the major ways to distinguish between types of exoplanets. The categories used were Earth-like, Super-Earth, Neptune-like and Gas-Giant. The planet radius vs host-star temperature graph shows that larger planets appear more commonly around hotter stars, but there is not a strong trend to determine a planet's radius.

## Machine Learning

### Features and Targets

The model's aim is to predict the planet's radius. The inputs were observed transit depth, host-star radius, host-star temperature, orbital period, and planet mass for only one of the models. Only planets with a known radius were included so the observed radius could be compared after.

### Missing Data

Some of the measurements of the exoplanets were missing in the archive, so median imputation was used to fill the gaps in the data. The median was learned from the training data.

## Experiment: How Important was planet mass

Two models were used for this project, one trained with the planet's mass and one trained without to test the signifigance of the plane's mass in predictig is radius. The planet mass was pretty curcial to predicted the raduis of the planet and ipacted the accuracy of the predicted radii the model gave.
