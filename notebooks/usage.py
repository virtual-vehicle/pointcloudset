#!/usr/bin/env python
# coding: utf-8

# # Example usage of package
# First import the package, and pathlib which is required to handle files.
# 
# 
# Note on jupyter notebook useage: VS code can open them, but it still is buggy and you can lose data.
# Instead use jupyter lab, which is included in the docker image.
# 
# 
# Simply type:
# ```bash
# ju
# ```
# in the terminal, which starts a jupyter lab server.
# 

# In[ ]:


from pathlib import Path

import matplotlib.pyplot as plt

import lidar

plt.rcParams['figure.figsize'] = [20, 10]

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# Ignore the warning, which comes from the rospy package.

# ## Reading a ROS .bag file into the lidar.Dataset

# In[ ]:


testbag = Path().cwd().parent.joinpath("tests/testdata/test.bag")
testbag


# In[ ]:


testset = lidar.Dataset(testbag,lidar_name="ouster",keep_zeros=False)


# This reads the bagfile into the Dataset.
# Dataset only reads frames from the bagfile if needed, in order to save memory and make it possible to work which huge bagfiles.

# In[ ]:


print(testset)


# In[ ]:


len(testset)


# In order to see whats availble use "tab" to see the availble properties and methods. Alterantivly, use help(), dir(), and the documentation.
# Also shift tab is nice inside jupyter lab.

# In[ ]:


help(testset)


# In[ ]:


dir(testset)


# In[ ]:


testset.topics_in_bag


# You can also work diretly with the bag if needed.

# In[ ]:


type(testset.bag)


# In[ ]:


testset.end_time


# # Work with Frames
# 
# They are based on pandas dataframes and pyntcloud.
# This was necessary since, no pointcloud library currently support to store automotive lidar data which consists of more than just y,x,z and maybe R,G,B
# 
# First grab the first frame in the dataset:

# In[ ]:


testframe = testset[0]


# In[ ]:


print(testframe)


# Note that the number of points can vary, since all zero elements are deltede on import (see option keep_zero in the dataset)

# In[ ]:


len(testframe)


# ## Plotting
# Tip: move the mouse over the points to get detailed information

# In[ ]:


testframe.plot_interactive()


# This plot uses plotly as the backend, which can be rather time consuming. 
# WARNING: delte the output cells with plotly plots, they make the file very big.
# 
# Aternativly you can use:

# In[ ]:


testframe.plot_interactive(backend="pyntcloud")


# In[ ]:


testframe.plot_interactive(color="range", point_size=0.5)


# ## Working with pointcouds
# The frame consists maily of the properties "data" and "points".

# In[ ]:


testframe.data


# So data contains everything as a pandas dataframe. With all its power.

# In[ ]:


testframe.data.describe()


# In[ ]:


testframe.data.hist();


# Now a closer look a the points. 

# In[ ]:


testframe.points


# So its a Pyntcloud object https://pyntcloud.readthedocs.io/en/latest/PyntCloud.html which in turn is also based on Dataframes with many methods for pointclouds.
# In order to access the dataframe use this:

# In[ ]:


testframe.points.


# You can also work with the pointcloud with open3d

# In[ ]:


open3d_points = testframe.get_open3d_points()


# In[ ]:


open3d_points.get_max_bound()


# ## Pointcloud processing with build in methods
# Although you can do a lot with just data and points, on its own the Frame object has methods build in for processing, which in turn return a frame object. The use the power of dataframes, pyntcloud and open3d.
# 

# In[ ]:


newframe = testframe.limit("x",-5,5)


# In[ ]:


newframe.data.describe()


# So this is now a smaller Frame with x ranging from -5 to  5

# These command can also be chained together

# In[ ]:


newframe2 = testframe.limit("x",-5,5).limit("y",-5,5)


# In[ ]:


newframe2.data.describe()


# In[ ]:
