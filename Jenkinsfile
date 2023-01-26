pipeline {
    agent any
    environment {
        PROJECT_NAME = 'gym-super-mario-bros'
        DEPLOY_DIR_NAME = "${env.PROJECT_NAME}"
    }

    stages {
        // useless now, but leaving here
        stage('Add shebang for interactive shell...') {
            steps {
                sh '''#!/bin/bash -i
                '''
            }
        }
        stage('Build') {
            steps {
                echo "Building ${env.PROJECT_NAME} on ${env.JENKINS_URL}..."
                sh "mkdir ${env.DEPLOY_DIR_NAME}"

                // moves all files in a directory into a subdirectory
                // all the extra weirdness is there because just doing "mv *" will attempt to move the subdirectory
                // into itself, which gives you an error ...
                // solution was found here: https://stackoverflow.com/questions/4612157/how-to-use-mv-command-to-move-files-except-those-in-a-specific-directory
                sh "ls | grep -v ${DEPLOY_DIR_NAME} | xargs -t -I '{}' mv {} ${DEPLOY_DIR_NAME}"

                sh "tar -czvf ${env.PROJECT_NAME}.tar.gz ${env.DEPLOY_DIR_NAME}"
            }
        }

        stage('Deploy') {
            environment {
                DEPLOY_SERVER_URL = "olliejonas.com"
                DEPLOY_SERVER_USER = "root"

                TARGET_DIR = "/home/ollie/Projects/Dissertation/${PROJECT_NAME}"
                PYTHON_PACKAGES_DIR = "/home/ollie/Projects/python-packages"
                PYPI_DOCKER_CONTAINER_NAME = "PyPIServer"
            }
            steps {
                script {
                    env.DEPLOY_SERVER = "${env.DEPLOY_SERVER_USER}@${env.DEPLOY_SERVER_URL}"
                }
                echo "Deploying ${env.PROJECT_NAME} to ${env.DEPLOY_SERVER_URL}..."

                sshagent(credentials: ['projects']) {
                    sh """
                        [ -d ~/.ssh ] || mkdir ~/.ssh && chmod 0700 ~/.ssh
                        ssh-keyscan -t rsa,dsa ${DEPLOY_SERVER_URL} >> ~/.ssh/known_hosts
                        scp ${env.PROJECT_NAME}.tar.gz ${env.DEPLOY_SERVER}:${env.TARGET_DIR}

                        ssh -t -t ${env.DEPLOY_SERVER} << EOF
                        cd ${env.TARGET_DIR}
                        tar -xf ${env.DEPLOY_DIR_NAME}.tar.gz
                        cd ${env.DEPLOY_DIR_NAME}
                        python3 setup.py sdist
                        mv dist/* ${PYTHON_PACKAGES_DIR}
                        cd ..
                        rm -f ${env.DEPLOY_DIR_NAME}.tar.gz

                        docker stop ${PYPI_DOCKER_CONTAINER_NAME}
                        docker rm ${PYPI_DOCKER_CONTAINER_NAME}
                        docker run --name ${PYPI_DOCKER_CONTAINER_NAME} --detach --publish 8090:8080 --volume ${PYTHON_PACKAGES_DIR}:/data/packages pypiserver/pypiserver:latest
                        exit
                        EOF
                    """
                }
            }
        }
    }

    post {
        always {
            echo "Performing cleanup..."
            sh "rm -rf ${env.DEPLOY_DIR_NAME}"
            sh "rm -f ${env.PROJECT_NAME}.tar.gz"
        }
    }
}