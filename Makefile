CURR_BRANCH := $(shell git symbolic-ref --short HEAD)

deploy:
	@echo Pushing branch \"$(CURR_BRANCH)\" to Heroku...
	git push heroku $(CURR_BRANCH):main
	@echo Done.

run:
	docker-compose up

log:
	heroku logs -t
