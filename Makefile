install_packages:
	pip install --upgrade pip wheel
	pip install -r requirements.txt

run_api:
	uvicorn API.Api:app --reload

uninstall_dependencies :
	pip freeze | xargs pip uninstall -y

reinstall_dependencies : uninstall_dependencies install_packages

buil_docker_image :
	docker build -t local_image .

run_api_on_docker_image :
	echo "Starting Api on http://localhost:8080/"
	docker run  -p 8080:8000 local_image
