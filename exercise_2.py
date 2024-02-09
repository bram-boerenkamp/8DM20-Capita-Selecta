#%%
import elastix
import imageio
import os
import matplotlib.pyplot as plt
import numpy
import SimpleITK as sitk 

FOLDER_PATH = r'C:\Users\20192236\Documents\MY1\8DM20_CSinMIA'
CHEST_PATH = r'C:\Users\20192236\Documents\MY1\8DM20_CSinMIA\ImagesforPractical\chest_xrays'
ELASTIX_PATH = os.path.join(FOLDER_PATH, 'elastix.exe')
TRANSFORMIX_PATH = os.path.join(FOLDER_PATH, 'transformix.exe')



el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

fixed_image_path = os.path.join(CHEST_PATH, 'fixed_image.mhd')
moving_image_path = os.path.join(CHEST_PATH, 'moving_image.mhd')
parameter_file_path = os.path.join(CHEST_PATH, 'parameterswithpenalty.txt')

fixed_image = sitk.ReadImage(fixed_image_path)
moving_image = sitk.ReadImage(moving_image_path)
fixed_image = sitk.GetArrayFromImage(fixed_image)
moving_image = sitk.GetArrayFromImage(moving_image)

el.register(
    fixed_image=fixed_image_path,
    moving_image=moving_image_path,
    parameters=[parameter_file_path],
    output_dir=os.path.join(FOLDER_PATH, 'results'))



result = sitk.ReadImage(os.path.join(FOLDER_PATH, 'results', 'result.0.mhd'))
result = sitk.GetArrayFromImage(result)

tr =elastix.TransformixInterface(parameters=os.path.join(FOLDER_PATH, 'results', 'TransformParameters.0.txt'),
transformix_path=TRANSFORMIX_PATH)

tr.jacobian_determinant(output_dir=os.path.join(FOLDER_PATH, 'results'))


jac = sitk.ReadImage(os.path.join(FOLDER_PATH, 'results', 'SpatialJacobian.mhd'))
jac = sitk.GetArrayFromImage(jac)
out = jac>0

#plotting

fig, ax = plt.subplots(1, 4, figsize=(20, 5))
ax[0].imshow(fixed_image, cmap='gray')
ax[0].set_title('Fixed image')
ax[1].imshow(moving_image, cmap='gray')
ax[1].set_title('Moving image')
ax[2].imshow(result, cmap='gray')
ax[2].set_title('Transformed\nmoving image')
ax[3].imshow(out, cmap='gray')
plt.show()