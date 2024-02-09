#%%
import elastix
import imageio
import os
import matplotlib.pyplot as plt

FOLDER_PATH = r'C:\Users\20192236\Documents\MY1\8DM20_CSinMIA'
EXAMPLE_PATH = r'C:\Users\20192236\Documents\MY1\8DM20_CSinMIA\elastix-py-master\example_data'
ELASTIX_PATH = os.path.join(FOLDER_PATH, 'elastix.exe')
el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

fixed_image_path = os.path.join(FOLDER_PATH, 'patient1.jpg')
moving_image_path = os.path.join(FOLDER_PATH, 'patient2.jpg')
parameter_file_path = os.path.join(EXAMPLE_PATH, 'parameters_samplespace2_MR.txt')

el.register(
    fixed_image=fixed_image_path,
    moving_image=moving_image_path,
    parameters=[parameter_file_path],
    output_dir=os.path.join(FOLDER_PATH, 'results'))


result_path = os.path.join(FOLDER_PATH, 'results', 'result.0.tiff')
#plotting
fixed_image = imageio.imread(fixed_image_path)[:, :, 0]
moving_image = imageio.imread(moving_image_path)[:, :, 0]
transformed_moving_image = imageio.imread(result_path)

fig, ax = plt.subplots(2, 5, figsize=(20, 5))
ax[0,0].imshow(fixed_image, cmap='gray')
ax[0,0].set_title('Fixed image')
ax[0,1].imshow(moving_image, cmap='gray')
ax[0,1].set_title('Moving image')
ax[0,2].imshow(transformed_moving_image, cmap='gray')
ax[0,2].set_title('Transformed\nmoving image')
#.plot(log['itnr'], log['metric'])
for i in range(5):
    log = elastix.logfile(os.path.join(FOLDER_PATH, 'results', f'IterationInfo.0.R{i}.txt'))
    ax[1,i].plot(log['itnr'], log['metric'])
    ax[1,i].set_title(f'cost function of nr {i}')

plt.show()