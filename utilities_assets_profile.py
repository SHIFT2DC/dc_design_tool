import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta, date as date_obj


def add_noise_to_profile(profile: list, noise_std: float = 0.05) -> np.ndarray:
    """
    Ajoute du bruit gaussien à un profil de consommation.

    Args:
        profile (list): Profil de consommation agrégé (une liste de 24 valeurs).
        noise_std (float): Écart-type du bruit gaussien. Par défaut à 0.05.

    Returns:
        np.ndarray: Profil de consommation avec du bruit ajouté.
    """
    # Convertir le profil en un tableau numpy
    profile = np.array(profile)
    
    # Générer du bruit gaussien
    noise = np.random.normal(0, noise_std, size=profile.shape)
    
    # Ajouter le bruit au profil
    noisy_profile = profile + noise
    
    # S'assurer que les valeurs ne deviennent pas négatives
    noisy_profile = np.clip(noisy_profile, 0, None)
    
    return noisy_profile


def generate_computer_profiles(N: int, aggregated_profile: list) -> np.ndarray:
    """
    Génère N profils de consommation individuels pour des ordinateurs à partir d'un profil agrégé.

    Args:
        N (int): Nombre d'ordinateurs.
        aggregated_profile (list): Profil de consommation agrégé (une liste de 24 valeurs).

    Returns:
        np.ndarray: Un tableau de shape (N, 24) contenant les profils de consommation individuels.
    """
    # Convertir le profil agrégé en un tableau numpy
    aggregated_profile = np.array(aggregated_profile)
    
    # Initialiser un tableau pour stocker les profils individuels
    individual_profiles = np.zeros((N, 24))
    
    # Pour chaque heure, répartir la consommation entre les N ordinateurs
    for hour in range(24):
        # Générer des poids aléatoires pour chaque ordinateur
        weights = np.random.dirichlet(np.ones(N))  # S'assure que la somme des poids est 1
        # Répartir la consommation agrégée selon les poids
        individual_profiles[:, hour] = aggregated_profile[hour] * weights
    
    return individual_profiles


def plot_profiles(individual_profiles: np.ndarray, aggregated_profile: list, noisy_profile: np.ndarray = None) -> None:
    """
    Affiche les profils de consommation individuels, le profil agrégé et le profil bruité.

    Args:
        individual_profiles (np.ndarray): Profils individuels (shape: N x 24).
        aggregated_profile (list): Profil agrégé (une liste de 24 valeurs).
        noisy_profile (np.ndarray, optional): Profil agrégé avec du bruit ajouté.
    """
    hours = range(24)
    
    # Afficher les profils individuels
    for i in range(individual_profiles.shape[0]):
        plt.plot(hours, individual_profiles[i, :], label=f'Ordinateur {i+1}', alpha=0.5)
    
    # Afficher le profil agrégé
    plt.plot(hours, aggregated_profile, label='Profil agrégé', linewidth=2, color='black', linestyle='--')
    
    # Afficher le profil bruité (si fourni)
    if noisy_profile is not None:
        plt.plot(hours, noisy_profile, label='Profil agrégé bruité', linewidth=2, color='red', linestyle='-.')
    
    # Ajouter des labels et une légende
    plt.xlabel('Heures de la journée')
    plt.ylabel('Consommation (puissance normalisée)')
    plt.title('Profils de consommation des ordinateurs')
    plt.legend()
    plt.grid()
    plt.show()

# if __name__ == "__main__":
#     # Exemple d'utilisation
#     hours = range(24)
#     office_profile = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.5, 0.7, 0.8, 0.8, 0.8, 0.3, 0.3, 0.8, 0.8, 0.8, 0.7, 0.6, 0.2, 0.1, 0.1, 0.1]
#     # Ajouter du bruit au profil agrégé
#     noisy_office = add_noise_to_profile(office_profile, noise_std=0.05)
#     # Générer les profils individuels pour 5 ordinateurs
#     N = 10
#     individual_profiles = generate_computer_profiles(N, noisy_office)
#     # Afficher les profils
#     #plot_profiles(individual_profiles, office_profile, noisy_office)

#     HVAC_profile = [0.1, 0.1, 0.1, 0.5, 1, 1, 0.8, 0.7, 0.5, 0.3, 0.3, 0.3, 0.4, 0.7, 0.7, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
#     noisy_HVAC = add_noise_to_profile(HVAC_profile, noise_std=0.05)

#     EV_profile = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.9, 0.85, 0.8, 0.75, 0.7, 0.3, 0.25, 0.75, 0.7, 0.65, 0.6, 0.5, 0.2, 0.1, 0.1, 0.1]
#     noisy_EV = add_noise_to_profile(EV_profile, noise_std=0.05)

#     industrial_load_24h = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
#     noisy_industrial_load_24h = add_noise_to_profile(industrial_load_24h, noise_std=0.05)

#     industrial_load_12h = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0, 0, 0, 0]
#     noisy_industrial_load_12h = add_noise_to_profile(industrial_load_12h, noise_std=0.05)

#     plt.figure()
#     #plt.plot(hours,EV_profile,'g')
#     #plt.plot(hours,noisy_HVAC,'r')
#     #plt.plot(hours,noisy_office,'b')
#     #plt.plot(hours,noisy_HVAC,'k')
#     #plt.plot(hours,noisy_industrial_load_24h,'y')
#     plt.plot(hours,noisy_industrial_load_12h,'c')
#     plt.grid()
#     plt.show()


def add_noise_to_profile(profile: list, noise_std: float = 0.05) -> np.ndarray:
    """
    Ajoute du bruit gaussien à un profil de consommation.
    """
    profile = np.array(profile)
    noise = np.random.normal(0, noise_std, profile.shape)
    noisy_profile = profile + noise
    return np.clip(noisy_profile, 0, None)


def get_season(date: date_obj) -> str:
    """
    Détermine la saison en fonction du mois de la date.
    """
    month = date.month
    if 3 <= month <= 5:
        return 'spring'
    elif 6 <= month <= 8:
        return 'summer'
    elif 9 <= month <= 11:
        return 'autumn'
    else:
        return 'winter'


def get_day_type(date: date_obj, holidays: set) -> str:
    """
    Détermine si la date est un jour de semaine, un week-end ou des vacances.
    """
    if date in holidays:
        return 'vacation'
    elif date.weekday() >= 5:
        return 'weekend'
    else:
        return 'workday'


def interpolate_profile(base_profile: list, time_step_min: int) -> np.ndarray:
    """
    Interpole un profil de 24 heures à une résolution temporelle plus fine.
    """
    hours = np.arange(24)
    step_per_hour = 60 // time_step_min
    x_new = np.linspace(0, 23, 24 * step_per_hour)
    return np.interp(x_new, hours, base_profile)


def generate_load_profile(
    num_days: int,
    time_step_min: int = 60,
    base_profile: list = None,
    noise_std: float = 0.05,
    summer_coeficient=0.8,
    winter_coeficient=1.2,
    holiday_coefficient=1/5,
    weekend_coeficent=1/20,
    day_varation_sigma=0.1
) -> tuple:
    """
    Génère un profil de charge électrique sur une période donnée.

    Args:
        num_days (int): Nombre de jours pour lesquels générer le profil.
        time_step_min (int): Pas de temps en minutes (par défaut 60 minutes).
        base_profiles (dict): Profils de base pour chaque saison et type de jour.
        holidays (list): Liste des dates de vacances.
        noise_std (float): Écart-type du bruit gaussien.

    Returns:
        tuple: (timestamps, load_values, timestamp_indices) où :
            - timestamps est un tableau de datetime,
            - load_values est un tableau de valeurs de charge,
            - timestamp_indices est un tableau d'indices séquentiels.
    """
    # Date de début fixée au 01/01/2018
    start_date = date_obj(2018, 1, 1)
    end_date = start_date + timedelta(days=num_days - 1)
    holidays = [date_obj(2018, 7, x) for x in range(27,32)]+[date_obj(2018, 8, x) for x in range(1,20)]+[date_obj(2018, 12, x) for x in range(14,32)]

    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    timestamps = []
    load_values = []
    timestamp_indices = []
    current_index = 0

    holidays = set(holidays) if holidays else set()
    
    base_profile=np.array(base_profile)

    for date in dates:
        s=np.random.normal(1, day_varation_sigma, 1)[0]
        day_profile=base_profile*s
        season = get_season(date)
        day_type = get_day_type(date, holidays)
        
        if season == "summer":
            day_profile = day_profile*summer_coeficient
        elif season == "winter":
            day_profile = day_profile*winter_coeficient

        if day_type == 'vacation':
            day_profile = day_profile*holiday_coefficient
        if day_type == 'weekend':
            day_profile = day_profile*weekend_coeficent
        
        # Générer les timestamps pour la journée
        steps_per_day = 24 * 60 // time_step_min
        daily_timestamps = [
            datetime(date.year, date.month, date.day, (i * time_step_min) // 60, (i * time_step_min) % 60)
            for i in range(steps_per_day)
        ]
        
        # Générer les valeurs de charge
        if time_step_min == 60:
            daily_load = add_noise_to_profile(day_profile, noise_std)
            daily_load = np.clip(daily_load, 0, 1)
        else:
            interpolated = interpolate_profile(day_profile, time_step_min)
            noise = np.random.normal(0, noise_std, interpolated.shape)
            daily_load = np.clip(interpolated + noise, 0, 1)
        
        # Ajouter les timestamps, les valeurs de charge et les indices
        timestamps.extend(daily_timestamps)
        load_values.extend(daily_load)
        timestamp_indices.extend(range(current_index, current_index + steps_per_day))
        current_index += steps_per_day
    
    return np.array(timestamps), np.array(load_values), np.array(timestamp_indices)

# Exemple d'utilisation


if __name__ == "__main__":
    office_profile = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.3, 0.4, 0.5, 0.5, 0.5, 0.15, 0.15, 0.5, 0.5, 0.5, 0.35, 0.3, 0.15, 0.1, 0.1, 0.1]
    # Exemple de vacances
    holidays =[date_obj(2018, 7, x) for x in range(27,32)]+[date_obj(2018, 8, x) for x in range(1,20)]+[date_obj(2018, 12, x) for x in range(14,32)]
    # Générer un profil pour 10 jours avec un pas de temps de 30 minutes
    timestamps, load, indices = generate_load_profile(
        num_days=365,
        time_step_min=5,
        base_profile=office_profile,
        holidays=holidays,
        noise_std=0.01
    )
    
    # Affichage des 10 premiers éléments pour vérification
    '''print("Timestamps:", timestamps[:10])
    print("Load Values:", load[:10])
    print("Timestamp Indices:", indices[:10])'''
    
    # Affichage du profil généré
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(indices, load, label='Profil de charge')
    plt.xlabel('Temps')
    plt.ylabel('Consommation normalisée')
    plt.title('Profil de charge électrique sur 10 jours (pas de 30 minutes)')
    plt.grid(True)
    plt.legend()
    plt.show()
