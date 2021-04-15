Gitlab access to the docker registry
==========================================

On your very first time accessing our gitlab docker registry you need to do the following steps:

1.) generate an access token:
* in gitlab =&gt; User settings =&gt; Access token
* generate a token with &#34;read_registry&#34; rights
* store the token safely (KeePass)

2.) In the terminal login to our registry:

.. code-block:: bash

   docker login registry-gitlab.v2c2.at -u <vornamenachname> -p <token>