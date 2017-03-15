from sampleAPI import app as application

<VirtualHost *:5000>
	ServerName Firefly.com
	WSGIScriptAlias / ./
</VirtualHost>
