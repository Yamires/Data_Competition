# ========================================= Nouvelles features =========================================
    processed_df['gas_balance'] = processed_df['free_CO2'] / (processed_df['dissolved_oxygen'])
    processed_df['mineral_strength'] = processed_df['gypsum_level'] * processed_df['sodium']
    processed_df['fermentation_ratio'] = processed_df['fermentation_strength'] / (processed_df['alcohol_ABV'] )
    processed_df['acid_balance'] = processed_df['lactic_acid'] / (processed_df['pH']) # out 
    processed_df['acid_x_gas'] = processed_df['acid_balance'] * processed_df['gas_balance']
    processed_df['gravity_drop'] = processed_df['original_gravity'] - processed_df['final_gravity'] # out 
    processed_df['log_gravity_drop'] = np.log1p(processed_df['gravity_drop'].clip(lower=0))
    #processed_df = processed_df.drop(columns=['bitterness_IBU','diacetyl_concentration'], errors='ignore')

    #========================================= Legacy features =========================================
    processed_df['log_final_gravity'] = np.log1p(processed_df['final_gravity']) # out
    processed_df['acid_balance'] = processed_df['lactic_acid'] / (processed_df['pH'] + 1e-5) # out 
    processed_df['alcohol_efficiency'] = processed_df['alcohol_ABV'] / (processed_df['gravity_drop'] + 1e-5) # out 
    processed_df['residual_sugar_index'] = processed_df['final_gravity'] / (processed_df['original_gravity'] + 1e-5) # out 
    processed_df['lactic_intensity'] = processed_df['lactic_acid'] * processed_df['fermentation_strength'] # out 
    processed_df['acid_x_gravity'] = processed_df['acid_balance'] * processed_df['gravity_drop'] # out
    processed_df['sqrt_alcohol_eff'] = np.sqrt(processed_df['alcohol_efficiency'].clip(lower=0)) # out 
