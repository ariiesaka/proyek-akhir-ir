from django.shortcuts import render
from django.http import HttpResponse
import ir_datasets


def home(request):
    dataset = ir_datasets.load('antique')
    return HttpResponse(str(dataset.docs_iter()[0]))
