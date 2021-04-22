GitLab access to the docker registry
==========================================

On your very first time accessing our GitLab docker registry you need to do the following steps:

1.) Generate an access token:
   * In GitLab => User settings => Access token
   * Generate a token with &#34;read_registry&#34; rights
   * Store the token safely (KeePass)

2.) In the terminal login to our registry:

.. code-block:: bash

   docker login registry-gitlab.v2c2.at -u <firstnamelastname> -p <token>