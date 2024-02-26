#%%
import elastix
import imageio
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk

FOLDER_PATH = r'C:\Users\20192318\OneDrive - TU Eindhoven\8DM20'
EXAMPLE_PATH = r'C:\Users\20192318\OneDrive - TU Eindhoven\8DM20\example_data'
Result1_PATH = r'C:\Users\20192318\OneDrive - TU Eindhoven\8DM20\Res1'
Result2_PATH = r'C:\Users\20192318\OneDrive - TU Eindhoven\8DM20\Res2'
ELASTIX_PATH = os.path.join(Result1_PATH, 'elastix.exe')
el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

fixed_image_path = os.path.join(EXAMPLE_PATH, 'patient1.jpg')
moving_image_path1 = os.path.join(EXAMPLE_PATH, 'patient2.jpg')
parameter_file_path = os.path.join(EXAMPLE_PATH, 'GeneralParameterAffine.txt')

itk_image = sitk.ReadImage('fixed_image.mhd')
fixed_image_path = sitk.GetArrayFromImage(itk_image)

itk_image = sitk.ReadImage('moving_image.mhd')
moving_image_path = sitk.GetArrayFromImage(itk_image)


el.register(
    fixed_image=fixed_image_path,
    moving_image=moving_image_path1,
    parameters=[parameter_file_path],
    output_dir=os.path.join(Result1_PATH, 'results'))


result1_path = os.path.join(Result1_PATH, 'results', 'result.0.tiff')

moving_image_path = result1_path
parameter_file_path = os.path.join(EXAMPLE_PATH, 'GeneralParameterBspline.txt')

el.register(
    fixed_image=fixed_image_path,
    moving_image=moving_image_path,
    parameters=[parameter_file_path],
    output_dir=os.path.join(Result2_PATH, 'results'))

result2_path = os.path.join(Result2_PATH, 'results', 'result.0.tiff')
#plotting
fixed_image = imageio.imread(fixed_image_path)[:, :, 0]
moving_image = imageio.imread(moving_image_path1)[:, :, 0]
transformed_Affine = imageio.imread(result1_path)
transformed_Affine_Bspline = imageio.imread(result2_path)

fig, ax = plt.subplots(3, 5, figsize=(15, 10))
ax[0,0].imshow(fixed_image, cmap='gray')
ax[0,0].set_title('Fixed image')
ax[0,1].imshow(moving_image, cmap='gray')
ax[0,1].set_title('Moving image')
ax[0,2].imshow(transformed_Affine, cmap='gray')
ax[0,2].set_title('Transformed\nmoving imageAffine')
ax[0,3].imshow(transformed_Affine_Bspline, cmap='gray')
ax[0,3].set_title('Transformed\nmoving imageAffine+Bspline')
#.plot(log['itnr'], log['metric'])
for i in range(5):
    log = elastix.logfile(os.path.join(Result1_PATH, 'results', f'IterationInfo.0.R{i}.txt'))
    ax[1,i].plot(log['itnr'], log['metric'])
    ax[1,i].set_title(f'cost function of nr {i}')
for i in range(5):
    log = elastix.logfile(os.path.join(Result2_PATH, 'results', f'IterationInfo.0.R{i}.txt'))
    ax[2,i].plot(log['itnr'], log['metric'])
    ax[2,i].set_title(f'cost function of nr {i}')
plt.show()