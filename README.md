# oosi
out of sample inference (oosi)
20260703

How to install python developer environment for oosi via uv on Windows
(1)  Download and install Powershell:		https://aka.ms/install-powershell
(2)  Open Powershell and install uv: 		powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
(3)  With Powershell uv install python: 	uv python install 3.13
(4)  With Powershell uv create venv: 		uv venv venv-spyder
(5)  With Powershell activate venv: 		venv-spyder\Scripts\activate
(6)  With Powershell uv install packages: 	uv pip install ipywidgets lightgbm pandas scikit-learn spyder shapiq tabicl
(7)  With Powershell run Spyder: 			spyder
(8)  With Powershell uv create venv: 		uv venv venv-oosi
(9)  With Powershell activate venv: 		venv-oosi\Scripts\activate
(10) With Powershell uv install packages: 	uv pip install ipywidgets lightgbm openpyxl pyyaml seaborn shapiq spyder-kernels tabicl[shap]
(11) Optional if nvidia: 					uv pip install torch --index-url https://download.pytorch.org/whl/cu132 --upgrade
(12) In Spyder->Tools->Preferences->Python interpreter->Select interpreter navigate to venv-oosi/Scripts and select python.exe
(13) Configure config/oosi_1_mdl_configs_names.yaml
(14) Configure config/oosi_1_mdl_configs_<data_name>.yaml