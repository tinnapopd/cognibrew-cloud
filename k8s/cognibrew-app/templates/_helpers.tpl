{{/*
Expand the name of the chart.
*/}}
{{- define "cognibrew-app.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "cognibrew-app.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Chart name and version for the chart label.
*/}}
{{- define "cognibrew-app.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "cognibrew-app.labels" -}}
helm.sh/chart: {{ include "cognibrew-app.chart" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
{{- end }}

{{/*
Component-specific labels
*/}}
{{- define "cognibrew-app.componentLabels" -}}
{{ include "cognibrew-app.labels" .ctx }}
app.kubernetes.io/name: {{ .component }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{/*
Component-specific selector labels
*/}}
{{- define "cognibrew-app.componentSelectorLabels" -}}
app.kubernetes.io/name: {{ .component }}
app.kubernetes.io/instance: {{ .ctx.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{/*
Service fullnames
*/}}
{{- define "cognibrew-app.edge-gateway.fullname" -}}
{{- printf "%s-edge-gateway" (include "cognibrew-app.fullname" .) }}
{{- end }}

{{- define "cognibrew-app.vector-operation.fullname" -}}
{{- printf "%s-vector-operation" (include "cognibrew-app.fullname" .) }}
{{- end }}

{{- define "cognibrew-app.edge-sync.fullname" -}}
{{- printf "%s-edge-sync" (include "cognibrew-app.fullname" .) }}
{{- end }}
