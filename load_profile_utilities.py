import numpy as np
import matplotlib.pyplot as plt

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



if __name__ == "__main__":
    # Exemple d'utilisation
    hours = range(24)
    office_profile = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.5, 0.7, 0.8, 0.8, 0.8, 0.3, 0.3, 0.8, 0.8, 0.8, 0.7, 0.6, 0.2, 0.1, 0.1, 0.1]
    # Ajouter du bruit au profil agrégé
    noisy_office = add_noise_to_profile(office_profile, noise_std=0.05)
    # Générer les profils individuels pour 5 ordinateurs
    N = 10
    individual_profiles = generate_computer_profiles(N, noisy_office)
    # Afficher les profils
    #plot_profiles(individual_profiles, office_profile, noisy_office)

    HVAC_profile = [0.1, 0.1, 0.1, 0.5, 1, 1, 0.8, 0.7, 0.5, 0.3, 0.3, 0.3, 0.4, 0.7, 0.7, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    noisy_HVAC = add_noise_to_profile(HVAC_profile, noise_std=0.05)

    EV_profile = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.9, 0.85, 0.8, 0.75, 0.7, 0.3, 0.25, 0.75, 0.7, 0.65, 0.6, 0.5, 0.2, 0.1, 0.1, 0.1]
    noisy_EV = add_noise_to_profile(EV_profile, noise_std=0.05)

    industrial_load_24h = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    noisy_industrial_load_24h = add_noise_to_profile(industrial_load_24h, noise_std=0.05)

    industrial_load_12h = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0, 0, 0, 0]
    noisy_industrial_load_12h = add_noise_to_profile(industrial_load_12h, noise_std=0.05)

    plt.figure()
    #plt.plot(hours,EV_profile,'g')
    #plt.plot(hours,noisy_HVAC,'r')
    #plt.plot(hours,noisy_office,'b')
    #plt.plot(hours,noisy_HVAC,'k')
    #plt.plot(hours,noisy_industrial_load_24h,'y')
    plt.plot(hours,noisy_industrial_load_12h,'c')
    plt.grid()
    plt.show()