rl_df['gravity_drop'] = rl_df['original_gravity'] - rl_df['final_gravity']
rl_df['acid_balance'] = rl_df['lactic_acid'] / (rl_df['pH'] + 1e-5)
rl_df['gas_balance'] = rl_df['free_CO2'] / (rl_df['dissolved_oxygen'] + 1e-5)
rl_df['mineral_strength'] = rl_df['gypsum_level'] * rl_df['sodium']
rl_df['alcohol_efficiency'] = rl_df['alcohol_ABV'] / (rl_df['gravity_drop'] + 1e-5)
rl_df['residual_sugar_index'] = rl_df['final_gravity'] / (rl_df['original_gravity'] + 1e-5)
rl_df['fermentation_ratio'] = rl_df['fermentation_strength'] / (rl_df['alcohol_ABV'] + 1e-5)
rl_df['lactic_intensity'] = rl_df['lactic_acid'] * rl_df['fermentation_strength']
