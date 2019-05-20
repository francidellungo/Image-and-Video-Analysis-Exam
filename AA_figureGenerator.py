import numpy as np
import matplotlib.pyplot as plt
import json
import mxnet as mx

# class NDArrayEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, mx.nd.NDArray):
#             return obj.asnumpy().tolist()
#         return json.JSONEncoder.default(self, obj)

colours = [ 'b', 'g', 'y', 'c', 'k', 'm' ]


def saveFigures(methods, scores_names, all_performances, measures, pickle_base, path_figs, thresholds, scale):

        all_performances = np.array(all_performances)

        print(all_performances)
        print(all_performances.shape)

        for j, title  in enumerate(scores_names):
                saveThreshFigure(methods, measures, [ measure[j] for measure in all_performances ], path_figs, title, thresholds)
                # for j, title  in enumerate(scores_names):
                saveROCScoreFigure(methods, measures, [ measure[j] for measure in all_performances ], title, path_figs)

        for i, (cod, (measure, mea))  in enumerate(measures):
                saveROCMeasureFigure(methods, scores_names, measure, all_performances[i], path_figs)



def saveThreshFigure(methods, measures, measure_on_score, path_figs, title, thresholds):

        x = np.linspace(0, 1.0, len(measure_on_score[0]))

        # print(len(measure_on_score))
        EERs = dict()

        for i,  (_, (measure, mea))  in enumerate(measures):
                performance_params = measure_on_score[i]

                # print(len(performance_params))
                y_FRR = performance_params[:,1]

                y_FARs = [      performance_params[:,2],
                                performance_params[:,3],
                                performance_params[:,4]]

                # yp_FAR = performance_params[:,2]
                # yc_FAR = performance_params[:,3]
                # ym_FAR = performance_params[:,4]
                # print(len(y_FAR))
                EER = dict()

                for i, t in enumerate(['p', 'c', 'm']):
                        idx_EER = np.argmin(np.abs(np.subtract(y_FARs[i], y_FRR)))
                        print(y_FARs[i][idx_EER])
                        a = np.array(y_FARs[i][idx_EER])
                        EER[t] = [x[idx_EER], a.tolist()]

                EERs[mea] = EER 
                
                plt.plot(x, y_FRR, 'r', label='FRR')

                for i, (t, meth) in enumerate(methods):
                        plt.plot(x, y_FARs[i], colours[i], label='FAR'+meth )
                        plt.scatter([EER[t][0]], [EER[t][1]], c=colours[len(colours)-i-1] , label='EER'+meth)

                plt.xlabel('Threshold')
                plt.ylabel('Probability')
                plt.title('Finding best Threshold - ' + measure)
                plt.legend()
                plt.savefig(path_figs + 'thresh_fusion_' + measure + '.png')
                plt.close()

        print(EERs)

        with open(thresholds + 'thresholds_' + title + '.json', 'w') as f:
                # json_str = json.dumps(EERs, cls=NDArrayEncoder, indent=4)
                json.dump(EERs, f)


def saveROCScoreFigure(methods, measures, measure_on_score, title, path_figs):

        for i, (_, (measure, _))  in enumerate(measures):
                performance_params = measure_on_score[i]
                y_TAR = performance_params[:,0]
                
                y_FARs = [      performance_params[:,2],
                                performance_params[:,3],
                                performance_params[:,4]]
                
                for i, (t, meth) in enumerate(methods):
                        plt.plot(y_FARs[i], y_TAR, colours[i], label=meth )
                        
        
                plt.xlabel('FAR')
                plt.ylabel('TAR')
                plt.xscale('log')
                plt.title('ROC - '+ title + ' ' + measure )
                plt.legend()
                plt.savefig( path_figs + 'ROC_'+ title + '_' + measure + '.png')
                plt.close()

        for i, (_, (measure, _))  in enumerate(measures):
                performance_params = measure_on_score[i]
                y_TAR = performance_params[:,0]
                y_FAR = performance_params[:,4]
                plt.plot(y_FAR, y_TAR, colours[i], label=measure )
                        
        
        plt.xlabel('FAR')
        plt.ylabel('TAR')
        plt.xscale('log')
        plt.title('ROC - '+ title +'(with meanshape)' )
        plt.legend()
        plt.savefig( path_figs + 'ROC_'+ title +'_meanshape.png')
        plt.close()

        


def saveROCMeasureFigure(methods, scores_names, measure, row_score, path_figs ):

        for j, title in enumerate(scores_names):
                performance_params = row_score[j]
                y_TAR = performance_params[:,0]
                
                y_FARs = [      performance_params[:,2],
                                performance_params[:,3],
                                performance_params[:,4]]
                for i, (t, meth) in enumerate(methods):
                        plt.plot(y_FARs[i], y_TAR, colours[i], label=meth )
        
                plt.xlabel('FAR')
                plt.ylabel('TAR')
                plt.xscale('log')
                plt.title('ROC - '+measure+' '+title)
                plt.legend()
                plt.savefig( path_figs + 'ROC_'+ measure +'_'+ title +'.png')
                plt.close()

        for j, title in enumerate(scores_names):
                performance_params = row_score[j]
                y_TAR = performance_params[:,0]
                y_FAR = performance_params[:,4]
                plt.plot(y_FAR, y_TAR, colours[j], label=title )

        plt.xlabel('FAR')
        plt.ylabel('TAR')
        plt.xscale('log')
        plt.title('ROC - '+measure+'(with meanshape)')
        plt.legend()
        plt.savefig( path_figs + 'ROC_'+ measure +'_meanshape.png')
        plt.close()
