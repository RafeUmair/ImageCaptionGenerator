#Use Maven to build dependencies and app
FROM maven:3.9.6-eclipse-temurin-17 AS build

WORKDIR /app

#Copy only pom.xml first for dependency caching
COPY pom.xml .

#Download dependencies 
RUN mvn dependency:go-offline

#Copy source code after dependencies are cached
COPY src ./src

#Build the jar
RUN mvn -q package -DskipTests

# Use a minimal JRE for production
FROM eclipse-temurin:17-jre

WORKDIR /app

#Copy the jar from the build stage
COPY --from=build /app/target/*.jar app.jar

EXPOSE 8080

#Run the application
CMD ["java", "-jar", "app.jar"]