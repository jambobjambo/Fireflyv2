<VirtualHost *:5000>
	WSGIDaemonProcess sampleapp python-path=/var/www/FireflyNLP:/var/www/FireflyNLP/fireflyENV/lib/python2.7/site-packages
	WSGIProcessGroup FireflyNLP
	WSGIScriptAlias / /var/www/FireflyNLP/sampleapp/wsgi.py
</VirtualHost>
