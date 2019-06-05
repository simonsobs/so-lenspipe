
-- mbs --
Simulate and save lensed alms
Characterize data -> noise templates

-- L3.1 --
Prepare filters
Calculate theory norm
Calculate MCMF
		  Generate sim and apply taper
		  Get lensing map from theory norm
		  Save mean
Calculate MCnorm
		  Generate sim and apply taper
		  Get lensing map from theory norm
		  Subtract MCMF
		  Cross with input
		  Divide by w3
		  Get (Clkk / mean cross power)
Calculate MCN1
Get data lensing map
		Get coadded inpainted CMB map and apply taper
		Get lensing map from theory norm
		Subtract MCMF
		Multiply by MCnorm
Calculate raw power
Save raw power
Calculate RDN0

Subtract RDN0 and MCN1





