from syncnet_functions import *

# Frontal
version = 'v4'

# # Multi-view
# version = 'v7'

# Mode = {'lip', 'audio', 'both'}
mode = 'lip'
# mode = 'audio'
# mode = 'both'

syncnet_model = load_pretrained_syncnet_model(version=version, mode=mode, verbose=False)

