"""
label_generator.py
------------------
Phase 4: Build a dictionary mapping known merchants to categories.
This creates the initial labeled dataset from unlabeled transactions.
"""

import pandas as pd

# Comprehensive merchant-to-category mapping for French transactions
MERCHANT_TO_CATEGORY = {
    # === FOOD & DRINK ===
    # Groceries
    'FRANPRIX': 'Food & Drink / Groceries',
    'MARCHE U': 'Food & Drink / Groceries',
    'COURSES U': 'Food & Drink / Groceries',
    'SUPER U': 'Food & Drink / Groceries',
    'HYPER U': 'Food & Drink / Groceries',
    'AUCHAN': 'Food & Drink / Groceries',
    'CARREFOUR': 'Food & Drink / Groceries',
    'CARREFOUR CITY': 'Food & Drink / Groceries',
    'CARREFOUR MARKET': 'Food & Drink / Groceries',
    'LIDL': 'Food & Drink / Groceries',
    'ALDI': 'Food & Drink / Groceries',
    'MONOPRIX': 'Food & Drink / Groceries',
    'CASINO': 'Food & Drink / Groceries',
    'G20': 'Food & Drink / Groceries',
    'PROXI': 'Food & Drink / Groceries',
    'SPAR': 'Food & Drink / Groceries',
    'INTERMARCHE': 'Food & Drink / Groceries',
    'CORA': 'Food & Drink / Groceries',
    'E.LECLERC': 'Food & Drink / Groceries',
    'BIO C BON': 'Food & Drink / Groceries',
    'LA VIE CLAIRE': 'Food & Drink / Groceries',
    'BIOCOOP': 'Food & Drink / Groceries',
    'NATURALIA': 'Food & Drink / Groceries',
    'GRAND FRAIS': 'Food & Drink / Groceries',
    'PICARD': 'Food & Drink / Groceries',

    # Fast Food
    'MCDONALDS': 'Food & Drink / Fast food',
    'KFC': 'Food & Drink / Fast food',
    'BURGER KING': 'Food & Drink / Fast food',
    'FIVE GUYS': 'Food & Drink / Fast food',
    'QUICK': 'Food & Drink / Fast food',
    'SUBWAY': 'Food & Drink / Fast food',
    'O TACOS': 'Food & Drink / Fast food',
    'PIZZA HUT': 'Food & Drink / Fast food',
    'DOMINOS PIZZA': 'Food & Drink / Fast food',
    'POKAWA': 'Food & Drink / Fast food',

    # Restaurants
    'FLUNCH': 'Food & Drink / Restaurants',
    'HIPPOPOTAMUS': 'Food & Drink / Restaurants',
    'BUFFALO GRILL': 'Food & Drink / Restaurants',
    'COURTEPAILLE': 'Food & Drink / Restaurants',
    'BIG MAMMA': 'Food & Drink / Restaurants',
    'MEZZO DI PASTA': 'Food & Drink / Restaurants',
    'EXKI': 'Food & Drink / Restaurants',
    'BAGELSTEIN': 'Food & Drink / Restaurants',
    'BCHEF': 'Food & Drink / Restaurants',

    # Coffee & Tea
    'STARBUCKS': 'Food & Drink / Coffee and tea',
    'PAUL': 'Food & Drink / Coffee and tea',
    'COLUMBUS CAFE': 'Food & Drink / Coffee and tea',
    'BRIOCHE DOREE': 'Food & Drink / Coffee and tea',
    'LA CROISSANTERIE': 'Food & Drink / Coffee and tea',

    # Food Delivery
    'DELIVEROO': 'Food & Drink / Food delivery',
    'UBER EATS': 'Food & Drink / Food delivery',
    'JUST EAT': 'Food & Drink / Food delivery',

    # === ENTERTAINMENT ===
    'NETFLIX': 'Entertainment / TV and movies',
    'DISNEY PLUS': 'Entertainment / TV and movies',
    'APPLE TV+': 'Entertainment / TV and movies',
    'CANAL+': 'Entertainment / TV and movies',
    'OCS': 'Entertainment / TV and movies',
    'PARAMOUNT+': 'Entertainment / TV and movies',
    'MOLOTOV': 'Entertainment / TV and movies',
    'DEEZER': 'Entertainment / Music',
    'SPOTIFY': 'Entertainment / Music',
    'STEAM': 'Entertainment / Video games',
    'PLAYSTATION': 'Entertainment / Video games',
    'XBOX': 'Entertainment / Video games',
    'NINTENDO ESHOP': 'Entertainment / Video games',
    'TWITCH': 'Entertainment / Video games',
    'TICKETMASTER': 'Entertainment / Other',
    'PATHE CINEMA': 'Entertainment / Other',
    'UGC': 'Entertainment / Other',
    'MK2': 'Entertainment / Other',
    'CGR CINEMAS': 'Entertainment / Other',
    'CINEMA GAUMONT': 'Entertainment / Other',

    # === TRANSPORTATION ===
    'ESSO': 'Transportation / Gas',
    'TOTALENERGIES': 'Transportation / Gas',
    'SHELL': 'Transportation / Gas',
    'BP': 'Transportation / Gas',
    'AVIA': 'Transportation / Gas',
    'AGIP': 'Transportation / Gas',
    'VINCI AUTOROUTES': 'Transportation / Tolls',
    'SANEF': 'Transportation / Tolls',
    'APRR': 'Transportation / Tolls',
    'PARKING INDIGO': 'Transportation / Parking',
    'Q-PARK': 'Transportation / Parking',
    'RATP': 'Transportation / Public Transport',
    'SNCF CONNECT': 'Transportation / Public Transport',
    'TER SNCF': 'Transportation / Public Transport',
    'OUIGO': 'Transportation / Public Transport',
    'TRAINLINE': 'Transportation / Public Transport',
    'FLIXBUS': 'Transportation / Public Transport',
    'ILE DE FRANCE MOBILITES': 'Transportation / Public Transport',
    'UBER': 'Transportation / Taxis and ride shares',
    'BOLT': 'Transportation / Taxis and ride shares',
    'HEETCH': 'Transportation / Taxis and ride shares',
    'BLABLACAR': 'Transportation / Taxis and ride shares',
    'LIME': 'Transportation / Taxis and ride shares',
    'DOTT': 'Transportation / Taxis and ride shares',
    'BLUELY': 'Transportation / Taxis and ride shares',
    'VELIB': 'Transportation / Taxis and ride shares',
    'NORAUTO': 'Transportation / Automobile maintenance and fees',

    # === GENERAL MERCHANDISE ===
    'UNIQLO': 'General Merchandise / Clothing',
    'H&M': 'General Merchandise / Clothing',
    'ZARA': 'General Merchandise / Clothing',
    'KIABI': 'General Merchandise / Clothing',
    'JULES': 'General Merchandise / Clothing',
    'CELIO': 'General Merchandise / Clothing',
    'PROMOD': 'General Merchandise / Clothing',
    'COCCINELLE': 'General Merchandise / Clothing',
    'PRINTEMPS': 'General Merchandise / Clothing',
    'GALERIES LAFAYETTE': 'General Merchandise / Clothing',
    'ZALANDO': 'General Merchandise / Clothing',
    'VINTED': 'General Merchandise / Clothing',
    'ACTION': 'General Merchandise / Clothing',
    'GIFI': 'General Merchandise / Clothing',
    'FNAC': 'General Merchandise / Electronics',
    'FNAC SPECTACLES': 'Entertainment / Other',
    'DARTY': 'General Merchandise / Electronics',
    'BOULANGER': 'General Merchandise / Electronics',
    'CDISCOUNT': 'General Merchandise / Electronics',
    'AMAZON': 'General Merchandise / Online marketplaces',
    'IKEA': 'General Merchandise / Houseware',
    'MAISONS DU MONDE': 'General Merchandise / Houseware',
    'CONFORAMA': 'General Merchandise / Houseware',
    'BUT': 'General Merchandise / Houseware',
    'CASTORAMA': 'General Merchandise / Houseware',
    'LEROY MERLIN': 'General Merchandise / Houseware',
    'BRICO DEPOT': 'General Merchandise / Houseware',
    'DECATHLON': 'General Merchandise / Sporting goods',
    'GO SPORT': 'General Merchandise / Sporting goods',
    'CULTURA': 'General Merchandise / Other',
    'KING JOUET': 'General Merchandise / Other',
    'NATURE ET DECOUVERTES': 'General Merchandise / Other',

    # === GENERAL SERVICES ===
    'BASIC-FIT': 'General Services / Health + Fitness',
    'FITNESS PARK': 'General Services / Health + Fitness',
    'ONYX GYM': 'General Services / Health + Fitness',
    'NEONESS': 'General Services / Health + Fitness',
    'MUTUELLE HARMONIE': 'General Services / Health insurance',
    'AXA': 'General Services / Other insurance',
    'ALLIANZ': 'General Services / Other insurance',
    'DIRECT ASSURANCE': 'General Services / Other insurance',
    'MAAF': 'General Services / Other insurance',
    'MACIF': 'General Services / Other insurance',
    'MATMUT': 'General Services / Other insurance',
    'GMF': 'General Services / Other insurance',
    'MAIF': 'General Services / Other insurance',
    'GROUPAMA': 'General Services / Other insurance',
    'LUKO': 'General Services / Other insurance',
    'DROPBOX': 'General Services / Cloud storage',
    'ADOBE': 'General Services / Other non-entertainment online subscriptions',
    'MICROSOFT 365': 'General Services / Other non-entertainment online subscriptions',
    'GOOGLE ONE': 'General Services / Other non-entertainment online subscriptions',
    'APPLE ICLOUD': 'General Services / Cloud storage',
    'NORDVPN': 'General Services / Other non-entertainment online subscriptions',
    'LINKEDIN PREMIUM': 'General Services / Other non-entertainment online subscriptions',
    'AMAZON PRIME': 'General Services / Other non-entertainment online subscriptions',
    'YOUBOOX': 'General Services / Other non-entertainment online subscriptions',
    'EKWATEUR': 'General Services / Home Repair + Maintenance',

    # === MEDICAL ===
    'PHARMACIE': 'Medical / Pharmacies and supplements',
    'PHARMACIE DU CENTRE': 'Medical / Pharmacies and supplements',
    'PHARMACIE LAFAYETTE': 'Medical / Pharmacies and supplements',
    'GRANDE PHARMACIE': 'Medical / Pharmacies and supplements',
    'PARAPHARMACIE': 'Medical / Pharmacies and supplements',
    'CERBALLIANCE': 'Medical / Pharmacies and supplements',
    'LABORATOIRE BIOGROUP': 'Medical / Pharmacies and supplements',
    'DENTAL CENTRE': 'Medical / Pharmacies and supplements',
    'CABINET MEDICAL': 'Medical / Pharmacies and supplements',
    'KINE CABINET': 'Medical / Pharmacies and supplements',
    'DOCTOLIB': 'Medical / Pharmacies and supplements',
    'ALAIN AFFLELOU': 'Medical / Pharmacies and supplements',
    'GRAND OPTICAL': 'Medical / Pharmacies and supplements',
    'GENERALE D OPTIQUE': 'Medical / Pharmacies and supplements',
    'KRYS': 'Medical / Pharmacies and supplements',
    'MARIONNAUD': 'Medical / Pharmacies and supplements',
    'NOCIBE': 'Medical / Pharmacies and supplements',
    'SEPHORA': 'Medical / Pharmacies and supplements',

    # === RENT & UTILITIES ===
    'ORANGE': 'Rent & Utilities / Mobile Phone',
    'SOSH': 'Rent & Utilities / Mobile Phone',
    'SFR': 'Rent & Utilities / Mobile Phone',
    'RED BY SFR': 'Rent & Utilities / Mobile Phone',
    'BOUYGUES TELECOM': 'Rent & Utilities / Mobile Phone',
    'FREE': 'Rent & Utilities / Mobile Phone',
    'LA POSTE MOBILE': 'Rent & Utilities / Mobile Phone',
    'LIVEBOX ORANGE': 'Rent & Utilities / Internet',
    'BOX FREE': 'Rent & Utilities / Internet',
    'VEOLIA EAU': 'Rent & Utilities / Water',
    'SUEZ EAU': 'Rent & Utilities / Water',
    'SAUR': 'Rent & Utilities / Water',
    'EDF': 'Rent & Utilities / Rent',
    'ENGIE': 'Rent & Utilities / Rent',
    'TOTALENERGIES ELEC': 'Rent & Utilities / Rent',

    # === BANK TRANSFERS ===
    'ATM WITHDRAWAL': 'Bank Transfers / ATM withdrawals',
    'BANK FEE': 'Bank Transfers / Other bank fees',
    'FOREIGN WITHDRAWAL FEE': 'Bank Transfers / Foreign transaction fees',
    'BANK OPPOSITION FEE': 'Bank Transfers / Other bank fees',
    'TRANSFER FEE': 'Bank Transfers / Other bank fees',
    'BANK INTERVENTION FEE': 'Bank Transfers / Other bank fees',
    'BANK AGIOS': 'Bank Transfers / Other bank fees',
    'CARD FEE': 'Bank Transfers / Other bank fees',

    # === INCOME ===
    'SALAIRE': 'Income / Other',
    'FRANCE TRAVAIL': 'Income / Other',
    'CAF': 'Income / Other',
    'PRIME': 'Income / Other',
    'REMBOURSEMENT CPAM': 'Income / Other',
    'REMBOURSEMENT MUTUELLE': 'Income / Other',
    'VIR RECU': 'Income / Other',

    # === OTHER / PAYMENT SERVICES ===
    'PAYPAL': 'General Services / Other',
    'REVOLUT': 'General Services / Other',
    'LYDIA': 'General Services / Other',
    'PAYLIB': 'General Services / Other',
    'WERO': 'General Services / Other',
    'SUMEUP': 'General Services / Other',
    'VIREMENT': 'General Services / Other',
}


def assign_category(merchant: str) -> str:
    """Assign category based on merchant name."""
    return MERCHANT_TO_CATEGORY.get(merchant, 'Unknown')


def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add generated_category column to dataframe."""
    df = df.copy()
    df['generated_category'] = df['extracted_merchant'].apply(assign_category)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_transactions.csv")
    df = generate_labels(df)
    print(f"Coverage: {(df['generated_category'] != 'Unknown').mean()*100:.1f}%")
    print(f"Categories: {df['generated_category'].nunique()}")
    print(df['generated_category'].value_counts())
    df.to_csv("data/processed/labeled_transactions.csv", index=False)
