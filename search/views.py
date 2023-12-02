from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from .bsbi import BSBIIndex
import ir_datasets


def home(request):
    return render(request, 'index.html')


def search(request):
    query = request.GET.get('query')

    if query is None:
        return redirect('home')

    page = int(request.GET.get('page', 0))
    docstore = ir_datasets.load('antique').docs_store()

    result = []
    start = page * 10
    end = start + 10
    BSBI_instance = BSBIIndex()

    docs = list(BSBI_instance.retrieve_bm25(query, k=100))

    for i, (score, doc) in enumerate(docs):
        if start <= i < end:
            content = docstore.get(doc).text
            result.append((doc.split('_')[0], content))

    response = {
        'query': query,
        'result': result,
        'pages': range(((len(docs) - 1) // 10) + 1),
        'empty': len(docs) < (start + 1)
    }

    return render(request, 'search.html', response)
