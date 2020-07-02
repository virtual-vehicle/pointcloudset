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


import lidar
from pathlib import Path


# Ignore the warning, which comes from the rospy package.

# ## Reading a ROS .bag file into the lidar.Dataset

# In[ ]:


testbag = Path().cwd().parent.joinpath("tests/testdata/test.bag")
testbag


# In[ ]:


testset = lidar.Dataset(testbag,lidar_name="ouster",keep_zeros=False)


# This reads the bagfile into the Dataset.
# Dataset only reads frames from the bagfile if needed, in order to save memory and make it possible to work which huge bagfiles.

# In[14]:


print(testset)


# In[15]:


len(testset)


# In order to see whats availble use "tab" to see the availble properties and methods. Alterantivly, use help(), dir(), and the documentation.

# In[17]:


help(testset)


# In[18]:


dir(testset)


# In[19]:


testset.topics_in_bag


# You can also work diretly with the bag if needed.

# In[21]:


type(testset.bag)


# # Work with Frames
# First grab the first frame in the dataset:

# In[22]:


testframe = testset[0]


# In[13]:


print(testframe)


# Note that the number of points can vary, since all zero elements are deltede on import (see option keep_zero in the dataset)

# In[ ]:


len(testframe)


# ## Plotting
# Tip: move the mouse over the points to get detailed information

# In[ ]:


testframe.plot_interactive()


# This plot uses plotly as the backend, which can be rather time consuming. Aternativly you can use:

# In[ ]:


testframe.plot_interactive(backend="pyntcloud")


# In[ ]:




