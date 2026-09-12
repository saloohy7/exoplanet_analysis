
# importing libraries 

import requests
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,pl_trandep,st_teff,st_rad,pl_masse,pl_rade,pl_orbper+from+ps+where+default_flag=1&format=json"
response = requests.get(url)
data = response.json()


# function for defining each type of planet based of radius

def get_planet_category(planet_radius):
        if planet_radius is not None:
            radius = planet_radius
            if radius < 1.5:
                return "Earth-like"
            elif 1.5 <= radius < 2.5:
                return "Super-Earth"
            elif 2.5 <= radius < 6:
                return "Neptune-like"
            else:
                return "Gas Giant"
        else:
            return "Unknown"

legend_elements = [
    Line2D([0], [0], marker="o", color="w", label="Earth-like", markerfacecolor="green", markersize=8),
    Line2D([0], [0], marker="o", color="w", label="Super-Earth", markerfacecolor="blue", markersize=8),
    Line2D([0], [0], marker="o", color="w", label="Neptune-like", markerfacecolor="orange", markersize=8),
    Line2D([0], [0], marker="o", color="w", label="Gas Giant", markerfacecolor="red", markersize=8)
]

#lists for temp, color, radii, and a list to group every planet type into a color 

temperatures = []
radii = []
colors = []
category_colors = {
    "Earth-like": "green",
    "Super-Earth": "blue",
    "Neptune-like": "orange",
    "Gas Giant": "red",
    "Unknown": "gray"
} 

#  calculate approximiate trasnit depth for each planet using radii of planet and star

for planet in data:
    temperature = planet.get("st_teff")
    planet_radius = planet.get("pl_rade")
    
    star_radius_raw = planet.get("st_rad") 
    if star_radius_raw and planet_radius:
        star_radius = star_radius_raw * 109.2
        transit_depth = ((planet_radius / star_radius) ** 2) * 100
    else:
        transit_depth = None
    real_depth = planet.get("pl_trandep")
    if transit_depth is not None and real_depth is not None:
        percent_error = ((transit_depth - real_depth) / real_depth) * 100 
    else:
        percent_error = None
    if planet_radius is not None:
        planet["category"] = get_planet_category(planet_radius)

    if temperature is not None and planet_radius is not None:
            temperatures.append(temperature)
            radii.append(planet_radius)
            color = category_colors[planet["category"]]
            colors.append(color)
    planet["transit_depth"] = transit_depth
    planet["percent_error"] = percent_error
    
# plot to compare different types of exoplanets with differernt temperatures of their host stars, color represent different planet types 

fig1, ax1 = plt.subplots(figsize=(10, 8))

ax1.scatter(temperatures, radii, c=colors, alpha=0.5)
ax1.set_xlabel("Host Star Temperature (K)")
ax1.set_ylabel("Planet Radius (Earth radii)")
ax1.set_xlim(0,13000)
ax1.set_ylim(0.1,35)
ax1.set_yscale("log")
ax1.legend(handles=legend_elements, title="Planet Category")
ax1.set_title("Exoplanet Radius vs Host Star Temperature")

# list used for model training, both for training with mass given and not given

X_with_mass = []
Y_with_mass = []    
X_without_mass =[]
Y_without_mass =[]
ml_data = []

# adding all the data about the planet in empty list, and ignore any exoplanets with no pre-given radii

for planet in data:
    planet_radius = planet.get("pl_rade")

    if planet_radius is not None:
         transit_depth = planet.get("pl_trandep")
         star_radius = planet.get("st_rad")
         temperature = planet.get("st_teff")
         orbital_period = planet.get("pl_orbper")
         mass = planet.get("pl_masse")
         ml_data.append([transit_depth,star_radius,temperature,orbital_period,mass,planet_radius])            



for ml_planet in ml_data:
  X_with_mass.append(ml_planet[:-1])
  Y_with_mass.append(ml_planet[-1])


# model training with the mass included, to test accuracy of the model, missing values are replaced using median of training data

X_train,X_test,Y_train,Y_test = train_test_split(X_with_mass,Y_with_mass,test_size=0.2,random_state=40)   
imputer_with_mass = SimpleImputer(strategy="median")

X_train = imputer_with_mass.fit_transform(X_train)
X_test = imputer_with_mass.transform(X_test)


        
model_with_mass = RandomForestRegressor(n_estimators=100,random_state=40)
model_with_mass.fit(X_train,Y_train)
predictions_with_mass = model_with_mass.predict(X_test)

mae_with_mass = mean_absolute_error(Y_test,predictions_with_mass)
r2_with_mass = r2_score(Y_test,predictions_with_mass)

print("Mean Absolute Error with mass:", mae_with_mass)
print("R² Score with mass:", r2_with_mass)
    


for ml_planet in ml_data:
  X_without_mass.append(ml_planet[:-2])
  Y_without_mass.append(ml_planet[-1])

# same model repeated but without the mass of the exoplanmet given to see how much accuracy has changed, same test and split for fair training

X1_train,X1_test,Y1_train,Y1_test = train_test_split(X_without_mass,Y_without_mass,test_size=0.2,random_state=40)   
imputer_without_mass = SimpleImputer(strategy="median")

X1_train = imputer_without_mass.fit_transform(X1_train)
X1_test = imputer_without_mass.transform(X1_test)


        
model1 = RandomForestRegressor(n_estimators=100,random_state=40)
model1.fit(X1_train,Y1_train)
predictions_without_mass = model1.predict(X1_test)

mae_without_mass = mean_absolute_error(Y1_test,predictions_without_mass)
r2_without_mass = r2_score(Y1_test,predictions_without_mass)

print("Mean Absolute Error one:", mae_without_mass)
print("R² Score one:", r2_without_mass)
        
# comparing both models predicted radii against observed raddi using graphs
   
fig2, ax2 = plt.subplots(figsize=(8, 8))

ax2.scatter(Y_test, predictions_with_mass, alpha=0.5,color = "blue",label="Predictions with mass")

min_radius = min(Y_test)
max_radius = max(Y_test)

ax2.plot(
    [min_radius, max_radius],
    [min_radius, max_radius],
    linestyle="--",
    label="Perfect prediction"
)

ax2.set_xlim(0,55)
ax2.set_ylim(0,55)
ax2.set_xlabel("Observed Planet Radius (Earth radii)")
ax2.set_ylabel("Predicted Planet Radius (Earth radii)")
ax2.set_title("Random Forest Radius Predictions — With Planet Mass")

ax2.text(
    0.05,0.95,
    f"MAE = {mae_with_mass:.2f} Earth radii\nR² = {r2_with_mass:.3f}",
    transform=ax2.transAxes,
    verticalalignment="top"
)

ax2.legend()


fig3, ax3 = plt.subplots(figsize=(8, 8))

ax3.scatter(Y1_test, predictions_without_mass, alpha=0.5,color ="orange",label="Predictions without mass")

min_radius = min(Y1_test)
max_radius = max(Y1_test)

ax3.plot(
    [min_radius, max_radius],
    [min_radius, max_radius],
    linestyle="--",
    label="Perfect prediction"
)

ax3.set_xlim(0,55)
ax3.set_ylim(0,55)
ax3.set_xlabel("Observed Planet Radius (Earth radii)")
ax3.set_ylabel("Predicted Planet Radius (Earth radii)")
ax3.set_title("Random Forest Radius Predictions — Without Planet Mass")

ax3.text(
    0.05,0.95,
    f"MAE = {mae_without_mass:.2f} Earth radii\nR² = {r2_without_mass:.3f}",
    transform=ax3.transAxes,
    verticalalignment="top"
)

ax3.legend()

plt.show()
